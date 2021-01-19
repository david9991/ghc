{-# LANGUAGE CPP #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}

-- | Types for the Constructed Product Result lattice. "GHC.Core.Opt.CprAnal"
-- and "GHC.Core.Opt.WorkWrap.Utils" are its primary customers via 'GHC.Types.Id.idCprInfo'.
module GHC.Types.Cpr (
    TerminationFlag (Terminates),
    Cpr, topCpr, conCpr, whnfTermCpr, divergeCpr, lubCpr, asConCpr,
    CprType (..), topCprType, whnfTermCprType, lubCprType, lubCprTypes,
    pruneDeepCpr, splitConCprTy, applyCprTy, abstractCprTy,
    abstractCprTyNTimes, ensureCprTyArity, trimCprTy,
    forceCprTy, forceCpr, bothCprType,
    cprTransformDataConWorkSig, cprTransformSig, argCprTypesFromStrictSig,
    CprSig (..), mkCprSig, mkCprSigForArity,
    topCprSig, seqCprSig
  ) where

#include "HsVersions.h"

import GHC.Prelude

import GHC.Types.Basic
import GHC.Types.Demand
import GHC.Types.Unbox
import GHC.Core.DataCon
import GHC.Core.Type
import GHC.Core.Multiplicity
import GHC.Utils.Binary
import GHC.Utils.Misc
import GHC.Utils.Outputable
import GHC.Utils.Panic
import qualified Data.Semigroup as Semigroup
import Control.Monad.Trans.Writer.CPS
import Control.Monad (zipWithM, zipWithM_)

-- import GHC.Driver.Ppr

---------------
-- * KnownShape

data KnownShape r
  = BotSh
  | ConSh !ConTag [r]
  | TopSh
  deriving Eq

seqKnownShape :: (r -> ()) -> KnownShape r -> ()
seqKnownShape seq_r (ConSh _ args) = foldr (seq . seq_r) () args
seqKnownShape _     _              = ()

lubKnownShape :: (r -> r -> r) -> KnownShape r -> KnownShape r -> KnownShape r
lubKnownShape _     BotSh            sh               = sh
lubKnownShape _     sh               BotSh            = sh
lubKnownShape _     TopSh            _                = TopSh
lubKnownShape _     _                TopSh            = TopSh
lubKnownShape lub_r (ConSh t1 args1) (ConSh t2 args2)
  | t1 == t2, args1 `equalLength` args2
  = (ConSh t1 (zipWith lub_r args1 args2))
  | otherwise
  = TopSh

pruneKnownShape :: (Int -> r -> r) -> Int -> KnownShape r -> KnownShape r
pruneKnownShape _       0     _              = TopSh
pruneKnownShape prune_r depth (ConSh t args) = ConSh t (map (prune_r (depth - 1)) args)
pruneKnownShape _       _     sh             = sh

----------------
-- * Termination

data TerminationFlag
  = Terminates
  | MightDiverge
  deriving Eq

lubTermFlag :: TerminationFlag -> TerminationFlag -> TerminationFlag
lubTermFlag MightDiverge _            = MightDiverge
lubTermFlag _            MightDiverge = MightDiverge
lubTermFlag Terminates   Terminates   = Terminates

data Termination
  = Term_ !TerminationFlag !(KnownShape Termination)
  -- ^ Don't use 'Term_', use 'Term' instead! Otherwise the derived Eq instance
  -- is broken.
  deriving Eq

-- | Normalises the nested termination info according to
-- > TopSh === ConSh t [topTerm..]
-- > BotSh === ConSh t [botTerm..]
-- (Note that this identity doesn't hold for CPR!):
normTermShape :: KnownShape Termination -> KnownShape Termination
normTermShape (ConSh _ fields)
  | all (== topTerm) fields = TopSh
  | all (== botTerm) fields = BotSh
normTermShape sh            = sh

pattern Term :: TerminationFlag -> KnownShape Termination -> Termination
pattern Term tf s <- (Term_ tf s)
  where
    -- 'normTermShape' is main point of this synonym. The first 4 case alts
    -- are only for interning purposes.
    Term tf (normTermShape -> sh) = case (tf, sh) of
      (Terminates,   BotSh) -> botTerm
      (Terminates,   TopSh) -> whnfTerm
      (MightDiverge, BotSh) -> divergeTerm
      (MightDiverge, TopSh) -> topTerm
      (tf,           sh   ) -> Term_ tf sh
{-# COMPLETE Term #-}

topTerm :: Termination
topTerm = Term_ MightDiverge TopSh

botTerm :: Termination
botTerm = Term_ Terminates BotSh

whnfTerm :: Termination
whnfTerm = Term_ Terminates TopSh

divergeTerm :: Termination
-- Just because we intern all the other combinations
divergeTerm = Term_ MightDiverge BotSh

lubTerm :: Termination -> Termination -> Termination
lubTerm (Term tf1 sh1) (Term tf2 sh2) =
  Term (lubTermFlag tf1 tf2) (lubKnownShape lubTerm sh1 sh2)

pruneDeepTerm :: Int -> Termination -> Termination
pruneDeepTerm depth (Term tf sh) =
  Term tf (pruneKnownShape pruneDeepTerm depth sh)

seqTerm :: Termination -> ()
seqTerm (Term _ l) = seqKnownShape seqTerm l

-- Reasons for design:
--  * We want to share between Cpr and Termination, so KnownShape
--  * Cpr is different from Termination in that we give up once one result
--    isn't constructed
--  * These are the key values to support, in case of a redesign. Write them down first:
--      topTerm, botTerm, whnfTerm, topCpr, botCpr, conCpr
--  * That is: For Termination we might or might not have nested info,
--    independent of termination of the current level. This is why Maybe
--    So, i.e. when we return a function (or newtype there-of) we'd have
--    something like @Termination Terminates Nothing@. We know evaluation
--    terminates, but we don't have any information on shape.
--    In fact, it's the same as
--  * Factoring Termination this way (i.e., TerminationFlag x shape) means less
--    duplication
-- Alternative: Interleave everything. Looks like this:
-- data Blub (b::Bool)
--   = NoCpr (Blub 'False)
--   | Cpr  ConTag TerminationFlag (Blub b)
--   | Term ConTag TerminationFlag (Blub b)
--   | TopBlub
--   | BotBlub
--  + More compact
--  + No Maybe (well, not here, still in Termination)
--  + Easier to handle in WW: Termination and Cpr encode compatible shape info
--    by construction
--  - Harder to understand: NoCpr means we can still have Termination info
--  - Spreads Termination stuff between two lattices
-- ... Probably not such a good idea, after all.
--
-- We keep TerminationFlag in Cpr if we can't transform beyond a MightDiverge anyway?
-- Because a seq might make a constructed product available again. WW makes sure to
-- split only as long as termination allows it, so we should be safe.

--------
-- * Cpr

data Cpr
  = Cpr_ !TerminationFlag !(KnownShape Cpr)
  | NoMoreCpr_ !Termination
  deriving Eq

pattern Cpr :: TerminationFlag -> KnownShape Cpr -> Cpr
pattern Cpr tf sh <- (Cpr_ tf sh)
  where
    Cpr Terminates   BotSh = botCpr
    Cpr Terminates   TopSh = whnfTermCpr
    Cpr MightDiverge BotSh = divergeCpr
    Cpr MightDiverge TopSh = topCpr
    Cpr tf           sh    = Cpr_ tf sh

pattern NoMoreCpr :: Termination -> Cpr
pattern NoMoreCpr t <- (NoMoreCpr_ t)
  where
    NoMoreCpr t
      | t == topTerm  = topCpr
      | t == whnfTerm = whnfTermCpr
      -- The following two would change CPR information:
      --- | t == botTerm  = botCpr
      --- | t == divergeTerm = divergeCpr
      | otherwise     = NoMoreCpr_ t
{-# COMPLETE Cpr, NoMoreCpr #-}

botCpr :: Cpr
botCpr = Cpr_ Terminates BotSh

topCpr :: Cpr
topCpr = Cpr_ MightDiverge TopSh

whnfTermCpr :: Cpr
whnfTermCpr = Cpr_ Terminates TopSh

-- | Used as
--
--   * The initial CPR of a recursive function in fixed-point iteration
--   * The CPR of 'undefined'/'error'/other sources of divergence.
--
-- We assume that evaluation to WHNF surely diverges (so 'MightDiverge'), but
-- are optimistic about all CPR and nested termination information. I.e., we
-- assume that returned tuple components terminate rapidly and construct a
-- product.
divergeCpr :: Cpr
divergeCpr = Cpr_ MightDiverge BotSh

conCpr :: ConTag -> [Cpr] -> Cpr
conCpr t fs = Cpr Terminates (ConSh t fs)

-- | Forget encoded CPR info, but keep termination info.
forgetCpr :: Cpr -> Termination
forgetCpr (NoMoreCpr t)         = t
forgetCpr (Cpr tf (ConSh t fs)) = Term tf (ConSh t (map forgetCpr fs))
forgetCpr (Cpr tf TopSh)        = Term tf TopSh
forgetCpr (Cpr tf BotSh)        = Term tf BotSh

lubCpr :: Cpr -> Cpr -> Cpr
lubCpr (Cpr tf1 sh1) (Cpr tf2 sh2)
  = Cpr (lubTermFlag tf1 tf2)
        (lubKnownShape lubCpr sh1 sh2)
lubCpr cpr1            cpr2
  = NoMoreCpr (lubTerm (forgetCpr cpr1) (forgetCpr cpr2))

-- | Trims CPR information (but keeps termination information) for e.g. thunks
trimCpr :: Cpr -> Cpr
trimCpr cpr@(Cpr _ BotSh) = cpr -- Don't trim away bottom, we still want to
                                -- unbox e.g. error thunks
trimCpr cpr               = NoMoreCpr (forgetCpr cpr)

pruneDeepCpr :: Int -> Cpr -> Cpr
pruneDeepCpr depth (Cpr tf sh)   = Cpr tf (pruneKnownShape pruneDeepCpr depth sh)
pruneDeepCpr depth (NoMoreCpr t) = NoMoreCpr (pruneDeepTerm depth t)

asConCpr :: Cpr -> Maybe (ConTag, [Cpr])
-- This is the key function consulted by WW
asConCpr (Cpr Terminates (ConSh t fields)) = Just (t, fields)
asConCpr _                                 = Nothing

seqCpr :: Cpr -> ()
seqCpr (Cpr _ l)     = seqKnownShape seqCpr l
seqCpr (NoMoreCpr t) = seqTerm t

------------
-- * CprType

-- | The abstract domain \(A_t\) from the original 'CPR for Haskell' paper.
data CprType
  = CprType
  { ct_arty :: !Arity
  -- ^ Number of value arguments the denoted expression eats before returning
  -- the 'ct_cpr'
  , ct_cpr  :: !Cpr
  -- ^ 'Cpr' eventually unleashed when applied to 'ct_arty' arguments
  }

instance Eq CprType where
  a == b =  ct_cpr a  == ct_cpr b
         && (ct_arty a == ct_arty b || isTopCprType a)

isTopCprType :: CprType -> Bool
isTopCprType (CprType _ cpr) = cpr == topCpr

topCprType :: CprType
topCprType = CprType 0 topCpr

botCprType :: CprType
botCprType = CprType 0 botCpr

whnfTermCprType :: CprType
whnfTermCprType = CprType 0 whnfTermCpr

lubCprType :: CprType -> CprType -> CprType
lubCprType ty1@(CprType n1 cpr1) ty2@(CprType n2 cpr2)
  | ct_cpr ty1 == botCpr && n1 <= n2 = ty2
  | ct_cpr ty2 == botCpr && n2 <= n1 = ty1
  -- There might be non-bottom CPR types with mismatching arities.
  -- Consider test DmdAnalGADTs. We want to return topCpr in these cases.
  -- Returning topCprType is a safe default.
  | n1 == n2
  = CprType n1 (lubCpr cpr1 cpr2)
  | otherwise
  = topCprType

lubCprTypes :: [CprType] -> CprType
lubCprTypes = foldl' lubCprType botCprType

extractArgCprAndTermination :: [CprType] -> [Cpr]
extractArgCprAndTermination = map go
  where
    go (CprType 0 cpr) = cpr
    -- we didn't give it enough arguments, so terminates rapidly
    go _               = whnfTermCpr

conCprType :: DataCon -> [CprType] -> CprType
conCprType dc args = CprType 0 (conCpr con_tag cprs)
  where
    con_tag = dataConTag dc
    cprs = addDataConTermination dc $ extractArgCprAndTermination args

addDataConTermination :: DataCon -> [Cpr] -> [Cpr]
-- See Note [No unboxed tuple for single, unlifted transit var]
addDataConTermination con cprs
  = zipWithEqual "addDataConTermination" add cprs strs
  where
    strs = dataConRepStrictness con
    add cpr str | isMarkedStrict str = snd $ forceCpr topSubDmd cpr -- like seq
                | otherwise          = cpr

splitConCprTy :: DataCon -> CprType -> Maybe [Cpr]
splitConCprTy dc (CprType 0 (Cpr _ l))
  | BotSh <- l
  = Just (replicate (dataConRepArity dc) botCpr)
  | (ConSh t fields) <- l
  , dataConTag dc == t
  , length fields == dataConRepArity dc -- See Note [CPR types and unsafeCoerce]
  = Just fields
splitConCprTy _  _
  = Nothing

applyCprTy :: CprType -> CprType
applyCprTy = applyCprTyNTimes 1

applyCprTyNTimes :: Arity -> CprType -> CprType
applyCprTyNTimes n (CprType m cpr)
  | m >= n        = CprType (m-n) cpr
  | cpr == botCpr = botCprType
  | otherwise     = topCprType

abstractCprTy :: CprType -> CprType
abstractCprTy = abstractCprTyNTimes 1

abstractCprTyNTimes :: Arity -> CprType -> CprType
abstractCprTyNTimes n ty@(CprType m cpr)
  | isTopCprType ty = topCprType
  | otherwise       = CprType (n+m) cpr

ensureCprTyArity :: Arity -> CprType -> CprType
ensureCprTyArity n ty@(CprType m _)
  | m == n    = ty
  | otherwise = topCprType

trimCprTy :: CprType -> CprType
trimCprTy (CprType arty cpr) = CprType arty (trimCpr cpr)

{- Note [CPR types and unsafeCoerce]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unsafe coercions may lead to looking into a CPR type analysed for a different
type than the type of the expression a case scrutinises. Example (like T16893):

  data T = T Int
  data U = U Int Int
  ... (case unsafeCoerce (U 1 2 :: U) of
        T i -> ...) ...

Although we can only derive bogus in these situations, we shouldn't try to
unpack the 2 field CPR type from the scrutinee into a 1 field type matching
the pattern. That led to a deliberate panic when calling @zipEqual@ in
'CprAnal.extendEnvForDataAlt'. Nothing bad was happening, just a precautionary
panic on T16893.
-}

----------------------------------------
-- * Forcing 'Termination' with 'Demand'
--
-- See Note [Rapid termination for strict binders]

-- | Abusing the Monoid instance of 'Semigroup.Any' to track a
-- 'TerminationFlag'.
newtype TerminationM a = TerminationM (Writer Semigroup.Any a)
  deriving (Functor, Applicative, Monad)

runTerminationM :: TerminationM a -> (TerminationFlag, a)
runTerminationM (TerminationM act) = case runWriter act of
  (a, Semigroup.Any True)  -> (MightDiverge, a)
  (a, Semigroup.Any False) -> (Terminates, a)

noteTermFlag :: TerminationFlag -> TerminationM ()
noteTermFlag MightDiverge = TerminationM (tell (Semigroup.Any True))
noteTermFlag Terminates   = TerminationM (tell (Semigroup.Any False))

-- | Forces possibly deep 'Termination' info of a 'CprType' according to
-- incoming 'ArgStr'. If there's any possibility that this 'MightDiverge',
-- return that.
-- See Note [Rapid termination for strict binders]
forceCprTy :: Demand -> CprType -> (TerminationFlag, CprType)
forceCprTy dmd ty = runTerminationM (idIfLazy forceCprTyM dmd ty)

-- | Lifts a 'TerminaionM' action from 'SubDemand's to 'Demand's by returning
-- the original argument if the 'Demand' is not strict.
-- Follows the rule "Don't force if lazy!".
idIfLazy :: (SubDemand -> cpr -> TerminationM cpr) -> Demand -> cpr -> TerminationM cpr
idIfLazy k (n :* sd) cpr
  | isStrict n = k sd cpr
  | otherwise  = return cpr

forceCprTyM :: SubDemand -> CprType -> TerminationM CprType
forceCprTyM sd ty
  | let arity = ct_arty ty
  , (m, body_sd) <- peelManyCalls arity sd
  , isStrict m
  , CprType body_arty body_cpr  <- applyCprTyNTimes arity ty
  = abstractCprTyNTimes arity . CprType body_arty <$> forceCprM body_sd body_cpr
  | otherwise
  = return ty

forceCprM :: SubDemand -> Cpr -> TerminationM Cpr
forceCprM sd (NoMoreCpr t)    = NoMoreCpr <$> forceTermM sd t
forceCprM sd (Cpr tf sh) = do
  -- 1. discharge head strictness by noting the term flag
  noteTermFlag tf
  -- 2. discharge *nested* strictness on available nested info
  sh' <- case (sh, sd) of
    (BotSh, _) -> return BotSh
    (ConSh t fields, viewProd (length fields) -> Just ds) | t == fIRST_TAG -> do
      fields' <- zipWithM (idIfLazy forceCprM) ds fields
      return (ConSh fIRST_TAG fields')
    (TopSh, Prod ds) -> do
      fields' <- mapM (flip (idIfLazy forceCprM) topCpr) ds
      return (ConSh fIRST_TAG fields')
    _ -> return sh -- just don't force anything
  return (Cpr Terminates sh')

forceTermM :: SubDemand -> Termination -> TerminationM Termination
forceTermM sd (Term tf sh) = do
  -- 1. discharge head strictness by noting the term flag
  noteTermFlag tf
  -- 2. discharge *nested* strictness on available nested info
  sh' <- case (sh, sd) of
    (BotSh, _) -> return BotSh
    (ConSh t fields, viewProd (length fields) -> Just ds) | t == fIRST_TAG -> do
      fields' <- zipWithM (idIfLazy forceTermM) ds fields
      return (ConSh fIRST_TAG fields')
    (TopSh, Prod ds) -> do
      fields' <- mapM (flip (idIfLazy forceTermM) topTerm) ds
      return (ConSh fIRST_TAG fields')
    _ -> return sh -- just don't force anything
  return (Term Terminates sh')

forceCpr :: SubDemand -> Cpr -> (TerminationFlag, Cpr)
forceCpr sd cpr = runTerminationM (forceCprM sd cpr)

-- | 'lubTerm's the given outer @TerminationFlag@ on the @CprType@s 'ct_term'.
bothCprType :: CprType -> TerminationFlag -> CprType
-- If tf = Terminates, it's just 'id'.
-- If tf = MightDiverge, it will only set the WHNF layer to MightDiverge,
-- leaving nested termination info (e.g. on product components) intact.
bothCprType ct Terminates   = ct
bothCprType ct MightDiverge = ct { ct_cpr = shallowDivCpr (ct_cpr ct) }

shallowDivCpr :: Cpr -> Cpr
shallowDivCpr (NoMoreCpr (Term _ sh)) = NoMoreCpr (Term MightDiverge sh)
shallowDivCpr (Cpr _ sh)              = Cpr MightDiverge sh

seqCprType :: CprType -> ()
seqCprType (CprType _ cpr) = seqCpr cpr

--------------
-- * CprSig

-- | The arity of the wrapped 'CprType' is the arity at which it is safe
-- to unleash. See Note [Understanding DmdType and StrictSig] in "GHC.Types.Demand"
newtype CprSig = CprSig { getCprSig :: CprType }
  deriving (Eq, Binary)

-- | Turns a 'CprType' computed for the particular 'Arity' into a 'CprSig'
-- unleashable at that arity. See Note [Understanding DmdType and StrictSig] in
-- "GHC.Types.Demand"
mkCprSigForArity :: Arity -> CprType -> CprSig
mkCprSigForArity arty = CprSig . ensureCprTyArity arty

topCprSig :: CprSig
topCprSig = CprSig topCprType

mkCprSig :: Arity -> Cpr -> CprSig
mkCprSig arty cpr = CprSig (CprType arty cpr)

seqCprSig :: CprSig -> ()
seqCprSig (CprSig sig) = seqCprType sig `seq` ()

-- | Get a 'CprType' for a 'DataCon', given 'CprType's for its fields.
cprTransformDataConWorkSig :: DataCon -> [CprType] -> CprType
-- What about DataCon *wrappers*? See Note [CPR for DataCon wrappers]
cprTransformDataConWorkSig con args
  | null (dataConExTyCoVars con)  -- No existentials
  , wkr_arity <= mAX_CPR_SIZE
  , args `lengthIs` wkr_arity
  -- , pprTrace "cprTransformDataConWorkSig" (ppr con <+> ppr wkr_arity <+> ppr args) True
  = abstractCprTyNTimes wkr_arity $ conCprType con args
  | otherwise
  = topCprType
  where
    wkr_arity = dataConRepArity con
    -- Note how we don't say 'MightDiverge' for returned CprType, although
    -- evaluation of unlifted args might in theory diverge. But
    --   * That's OK by the let/app invariant, which specifies that the
    --     things we forget to force are ok for speculation, so we should find
    --     that each argument 'Terminates' anyway.
    --   * Data constructor workers are in fact lazy! Evaluation is done by the
    --     (now inlined) wrapper.

    mAX_CPR_SIZE :: Arity
    mAX_CPR_SIZE = 10
    -- We do not treat very big tuples as CPR-ish:
    --      a) for a start we get into trouble because there aren't
    --         "enough" unboxed tuple types (a tiresome restriction,
    --         but hard to fix),
    --      b) more importantly, big unboxed tuples get returned mainly
    --         on the stack, and are often then allocated in the heap
    --         by the caller.  So doing CPR for them may in fact make
    --         things worse.

cprTransformSig :: StrictSig -> CprSig -> [CprType] -> CprType
-- See Note [Rapid termination for strict binders] in CprAnal
cprTransformSig str_sig (CprSig sig_ty) arg_tys
  | dmds <- argDmdsFromStrictSig str_sig
  , dmds `leLength` arg_tys
  , arg_tys `lengthIs` ct_arty sig_ty
  -- See Note [Rapid termination for strict binders]
  -- NB: 'dmds' doesn't account for strict fields, see
  -- Note [Add demands for strict constructors]. We don't have to, either: All
  -- forcing will be done by the call to the DataCon wrapper.
  , (tf, _) <- runTerminationM $ zipWithM_ (idIfLazy forceCprTyM) (dmds ++ repeat topDmd) arg_tys
  = -- pprTrace "cprTransformSig:ok" (ppr str_sig <+> ppr sig_ty <+> ppr arg_tys <+> ppr tf)
    sig_ty `bothCprType` tf
  | otherwise
  = -- pprTrace "cprTransformSig:topSig" (ppr str_sig <+> ppr sig_ty <+> ppr arg_tys)
    topCprType

-- | We have to be sure that 'cprTransformSig' and 'argCprTypesFromStrictSig'
-- agree in how they compute the 'Demand's for which the 'CprSig' is computed.
-- This function encodes the common (trivial) logic.
argDmdsFromStrictSig :: StrictSig -> [Demand]
argDmdsFromStrictSig = fst . splitStrictSig

-- | Produces 'CprType's the termination info of which match the given
-- strictness signature. Examples:
--
--   - A head-strict demand `S` would translate to `#`
--   - A tuple demand `S(S,L)` would translate to `#(#,*)`
--   - A call demand `C(S)` would translate to `. -> #(#,*)`
argCprTypesFromStrictSig :: UnboxingStrategy Demand -> [Type] -> StrictSig -> [CprType]
argCprTypesFromStrictSig want_to_unbox arg_tys sig
  = zipWith go arg_tys (argDmdsFromStrictSig sig)
  where
    go ty dmd
      | Unbox (DataConPatContext { dcpc_dc = dc, dcpc_tc_args = tc_args }) dmds
          <- want_to_unbox ty dmd
      -- No existentials; see Note [Which types are unboxed?])
      -- Otherwise we'd need to call dataConRepInstPat here and thread a UniqSupply
      , null (dataConExTyCoVars dc)
      , let arg_tys = map scaledThing (dataConInstArgTys dc tc_args)
      = conCprType dc (zipWith go arg_tys dmds)
      | otherwise
      = snd $ forceCprTy dmd topCprType

---------------
-- * Outputable

pprFields :: Outputable r => [r] -> SDoc
pprFields fs = parens (pprWithCommas ppr fs)

instance Outputable TerminationFlag where
  ppr MightDiverge = char '*'
  ppr Terminates   = char '#'

instance Outputable Termination where
  ppr (Term tf l) = ppr tf <> case l of
    TopSh -> empty
    BotSh -> text "(#..)"
    ConSh t fs -> ppr t <> pprFields fs

instance Outputable Cpr where
  ppr (NoMoreCpr t)          = ppr t
  -- I like it better without the special case
  -- ppr (Cpr MightDiverge Top) = empty
  -- ppr (Cpr Terminates   Bot) = char 'b'
  ppr (Cpr tf l)          = ppr tf <> case l of
    TopSh -> empty
    BotSh -> char 'b' -- Maybe just 'c'?
    ConSh t fs -> char 'c' <> ppr t <> pprFields fs

instance Outputable CprType where
  ppr (CprType arty cpr) = ppr arty <+> ppr cpr

-- | Only print the CPR result
instance Outputable CprSig where
  ppr (CprSig ty) = ppr (ct_cpr ty)

-----------
-- * Binary

instance Binary r => Binary (KnownShape r) where
  put_ bh BotSh   = putByte bh 0
  put_ bh (ConSh t fs) = putByte bh 1 *> putULEB128 bh t *> put_ bh fs
  put_ bh TopSh   = putByte bh 2
  get  bh = do
    h <- getByte bh
    case h of
      0 -> return BotSh
      1 -> ConSh <$> getULEB128 bh <*> get bh
      2 -> return TopSh
      _ -> pprPanic "Binary KnownShape: Invalid tag" (int (fromIntegral h))

instance Binary TerminationFlag where
  put_ bh Terminates   = put_ bh True
  put_ bh MightDiverge = put_ bh False
  get  bh = do
    b <- get bh
    if b
      then pure Terminates
      else pure MightDiverge

instance Binary Termination where
  put_ bh (Term tf l) = put_ bh tf >> put_ bh l
  get  bh = Term <$> get bh <*> get bh

instance Binary Cpr where
  put_ bh (Cpr tf l) = put_ bh True >> put_ bh tf >> put_ bh l
  put_ bh (NoMoreCpr t) = put_ bh False >> put_ bh t
  get  bh = do
    b <- get bh
    if b
      then Cpr <$> get bh <*> get bh
      else NoMoreCpr <$> get bh

instance Binary CprType where
  put_ bh (CprType args cpr) = do
    put_ bh args
    put_ bh cpr
  get  bh = CprType <$> get bh <*> get bh
