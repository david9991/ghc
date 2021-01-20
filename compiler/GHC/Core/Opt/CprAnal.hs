{-# LANGUAGE CPP #-}

-- | Constructed Product Result analysis. Identifies functions that surely
-- return heap-allocated records on every code path, so that we can eliminate
-- said heap allocation by performing a worker/wrapper split.
--
-- See https://www.microsoft.com/en-us/research/publication/constructed-product-result-analysis-haskell/.
-- CPR analysis should happen after strictness analysis.
-- See Note [Phase ordering].
module GHC.Core.Opt.CprAnal ( cprAnalProgram ) where

#include "HsVersions.h"

import GHC.Prelude

import GHC.Driver.Session
import GHC.Types.Demand
import GHC.Types.Cpr
import GHC.Types.Unbox
import GHC.Core
import GHC.Core.Opt.Arity ( splitFunNewTys )
import GHC.Core.Seq
import GHC.Utils.Outputable
import GHC.Types.Var.Env
import GHC.Types.Basic
import GHC.Types.Id
import GHC.Types.Id.Info
import GHC.Core.Utils   ( exprIsHNF, dumpIdInfoOfProgram, normSplitTyConApp_maybe )
import GHC.Core.TyCon
import GHC.Core.Type
import GHC.Core.FamInstEnv
import GHC.Core.Opt.WorkWrap.Utils
import GHC.Utils.Misc
import GHC.Utils.Error  ( dumpIfSet_dyn, DumpFormat (..) )
import GHC.Data.Maybe   ( isJust )

import Control.Monad ( guard )
import Data.List

-- import GHC.Driver.Ppr

{- Note [Constructed Product Result]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The goal of Constructed Product Result analysis is to identify functions that
surely return heap-allocated records on every code path, so that we can
eliminate said heap allocation by performing a worker/wrapper split
(via 'GHC.Core.Opt.WorkWrap.Utils.mkWWcpr_start').

@swap@ below is such a function:

  swap (a, b) = (b, a)

A @case@ on an application of @swap@, like
@case swap (10, 42) of (a, b) -> a + b@ could cancel away
(by case-of-known-constructor) if we "inlined" @swap@ and simplified. We then
say that @swap@ has the CPR property.

We can't inline recursive functions, but similar reasoning applies there:

  f x n = case n of
    0 -> (x, 0)
    _ -> f (x+1) (n-1)

Inductively, @case f 1 2 of (a, b) -> a + b@ could cancel away the constructed
product with the case. So @f@, too, has the CPR property. But we can't really
"inline" @f@, because it's recursive. Also, non-recursive functions like @swap@
might be too big to inline (or even marked NOINLINE). We still want to exploit
the CPR property, and that is exactly what the worker/wrapper transformation
can do for us:

  $wf x n = case n of
    0 -> case (x, 0) of -> (a, b) -> (# a, b #)
    _ -> case f (x+1) (n-1) of (a, b) -> (# a, b #)
  f x n = case $wf x n of (# a, b #) -> (a, b)

where $wf readily simplifies (by case-of-known-constructor and inlining @f@) to:

  $wf x n = case n of
    0 -> (# x, 0 #)
    _ -> $wf (x+1) (n-1)

Now, a call site like @case f 1 2 of (a, b) -> a + b@ can inline @f@ and
eliminate the heap-allocated pair constructor.

Note [Phase ordering]
~~~~~~~~~~~~~~~~~~~~~
We need to perform strictness analysis before CPR analysis, because that might
unbox some arguments, in turn leading to more constructed products.
Ideally, we would want the following pipeline:

1. Strictness
2. worker/wrapper (for strictness)
3. CPR
4. worker/wrapper (for CPR)

Currently, we omit 2. and anticipate the results of worker/wrapper.
See Note [CPR for binders that will be unboxed]
and Note [CPR in a DataAlt case alternative].
An additional w/w pass would simplify things, but probably add slight overhead.
So currently we have

1. Strictness
2. CPR
3. worker/wrapper (for strictness and CPR)
-}

--
-- * Analysing programs
--

cprAnalProgram :: DynFlags -> FamInstEnvs -> CoreProgram -> IO CoreProgram
cprAnalProgram dflags fam_envs binds = do
  let env            = emptyAnalEnv fam_envs
  let binds_plus_cpr = snd $ mapAccumL cprAnalTopBind env binds
  dumpIfSet_dyn dflags Opt_D_dump_cpr_signatures "Cpr signatures" FormatText $
    dumpIdInfoOfProgram (ppr . cprInfo) binds_plus_cpr
  -- See Note [Stamp out space leaks in demand analysis] in GHC.Core.Opt.DmdAnal
  seqBinds binds_plus_cpr `seq` return binds_plus_cpr

-- Analyse a (group of) top-level binding(s)
cprAnalTopBind :: AnalEnv
               -> CoreBind
               -> (AnalEnv, CoreBind)
cprAnalTopBind env (NonRec id rhs)
  = (env', NonRec id' rhs')
  where
    (id', rhs', env') = cprAnalBind TopLevel env noWidening id rhs

cprAnalTopBind env (Rec pairs)
  = (env', Rec pairs')
  where
    (env', pairs') = cprFix TopLevel env pairs

--
-- * Analysing expressions
--

-- | Analoguous to the abstract semantic function \(⟦_⟧ : Expr -> Env -> A\)
-- from "Constructed Product Result Analysis for Haskell"
cprAnal, cprAnal'
  :: AnalEnv
  -> [CprType]           -- ^ info about incoming arguments
  -> CoreExpr            -- ^ expression to be denoted by a 'CprType'
  -> (CprType, CoreExpr) -- ^ the updated expression and its 'CprType'

cprAnal env args e = -- pprTraceWith "cprAnal" (\res -> ppr (fst (res)) $$ ppr e) $
                     cprAnal' env args e

cprAnal' _ _ (Lit lit)     = (whnfTermCprType, Lit lit)
cprAnal' _ _ (Type ty)     = (whnfTermCprType, Type ty)      -- Doesn't happen, in fact
cprAnal' _ _ (Coercion co) = (whnfTermCprType, Coercion co)

cprAnal' env args (Var var) = (cprTransform env args var, Var var)

cprAnal' env args (Cast e co)
  = (cpr_ty, Cast e' co)
  where
    (cpr_ty, e') = cprAnal env args e

cprAnal' env args (Tick t e)
  = (cpr_ty, Tick t e')
  where
    (cpr_ty, e') = cprAnal env args e

cprAnal' env args (App fun (Type ty))
  = (fun_ty, App fun' (Type ty))
  where
    (fun_ty, fun') = cprAnal env args fun

cprAnal' env args (App fun arg)
  = (app_ty, App fun' arg')
  where
    (arg_ty, arg') = cprAnal env [] arg
    -- NB: arg_ty may have the CPR property. That is indeed important for data
    -- constructors.
    (fun_ty, fun') = cprAnal env (arg_ty:args) fun
    app_ty         = applyCprTy fun_ty
cprAnal' env args (Lam var body)
  | isTyVar var
  , (body_ty, body') <- cprAnal env args body
  = (body_ty, Lam var body')
  | otherwise
  = (lam_ty, Lam var body')
  where
    (arg_ty, body_args)
      | ty:args' <- args = (ty, args')      -- Info from e.g. a StrictSig or DataCon wrapper args
      | otherwise        = (topCprType, []) -- An anonymous lambda
    env'                 = extendSigEnv env var (CprSig arg_ty)
    (body_ty, body')     = cprAnal env' body_args body
    lam_ty               = abstractCprTy body_ty

cprAnal' env args (Case scrut case_bndr ty alts)
  = -- pprTrace "cprAnal:Case" (ppr scrut <+> ppr alt_tys <+> ppr res_ty) $
    (res_ty, Case scrut' case_bndr ty alts')
  where
    -- Analyse the scrutinee and additionally force the resulting CPR type with
    -- head strictness.
    (scrut_ty, scrut')        = cprAnal env [] scrut
    (whnf_flag, case_bndr_ty) = forceCprTy seqDmd scrut_ty
    (alt_tys, alts') = mapAndUnzip (cprAnalAlt env args case_bndr case_bndr_ty) alts
    res_ty           = lubCprTypes alt_tys `bothCprType` whnf_flag

cprAnal' env args (Let (NonRec id rhs) body)
  = (body_ty, Let (NonRec id' rhs') body')
  where
    (id', rhs', env') = cprAnalBind NotTopLevel env noWidening id rhs
    (body_ty, body')  = cprAnal env' args body

cprAnal' env args (Let (Rec pairs) body)
  = body_ty `seq` (body_ty, Let (Rec pairs') body')
  where
    (env', pairs')   = cprFix NotTopLevel env pairs
    (body_ty, body') = cprAnal env' args body

cprAnalAlt
  :: AnalEnv
  -> [CprType]      -- ^ info about incoming arguments
  -> Id             -- ^ case binder
  -> CprType        -- ^ info about the case binder
  -> Alt Var        -- ^ current alternative
  -> (CprType, Alt Var)
cprAnalAlt env args case_bndr case_bndr_ty (con,bndrs,rhs)
  = (rhs_ty, (con, bndrs, rhs'))
  where
    env_case_bndr = extendSigEnv env case_bndr (CprSig case_bndr_ty)
    env_alt
      -- See Note [CPR in a DataAlt case alternative]
      | DataAlt dc  <- con
      , Just fields <- splitConCprTy dc case_bndr_ty
      , let ids     = filter isId bndrs
      , let cpr_tys = map (CprSig . CprType 0) fields
      = extendSigEnvList env_case_bndr (zipEqual "cprAnalAlt" ids cpr_tys)
      | otherwise
      = env_case_bndr
    (rhs_ty, rhs') = cprAnal env_alt args rhs

--
-- * CPR transformer
--

cprTransform
  :: AnalEnv         -- ^ The analysis environment
  -> [CprType]       -- ^ info about incoming arguments
  -> Id              -- ^ The function
  -> CprType         -- ^ The demand type of the function
cprTransform env args id
  = -- pprTrace "cprTransform" (vcat [ppr id, ppr sig, ppr args])
    sig
  where
    sig
      -- Top-level binding, local let-binding, lambda arg or case binder
      | Just sig <- lookupSigEnv env id
      = cprTransformSig (idStrictness id) sig args
      -- See Note [CPR for data structures]
      | Just rhs <- cprDataStructureUnfolding_maybe id
      = fst $ cprAnal env args rhs
      -- See Note [CPR for DataCon wrappers]
      | isDataConWrapId id, let rhs = uf_tmpl (realIdUnfolding id)
      = fst $ cprAnal env args rhs
      -- Data constructor
      | Just con <- isDataConWorkId_maybe id
      = cprTransformDataConWorkSig con args
      -- Imported function or data con worker
      | isGlobalId id
      = cprTransformSig (idStrictness id) (idCprInfo id) args
      | otherwise
      = topCprType

--
-- * Analysing Bindings
--

--
-- ** Widening
--

type Widening = CprSig -> CprSig

noWidening :: Widening
noWidening = id

-- | A widening operator on 'CprSig' to ensure termination of fixed-point
-- iteration. See Note [Ensuring termination of fixed-point iteration]
depthWidening :: Widening
depthWidening (CprSig ty) = CprSig . prune . mark_diverging $ ty
  where
    mark_diverging ty = ty { ct_cpr = ct_cpr ty `lubCpr` divergeCpr }
    prune ty = ty { ct_cpr = pruneDeepCpr mAX_DEPTH (ct_cpr ty) }

-- This constant is quite arbitrary. We might well make it a CLI flag if needed
mAX_DEPTH :: Int
mAX_DEPTH = 4

--
-- ** Analysing a binding (one-round, the non-recursive case)
--

-- | Process the RHS of the binding for a sensible arity, add the CPR signature
-- to the Id, and augment the environment with the signature as well.
cprAnalBind
  :: TopLevelFlag
  -> AnalEnv
  -> Widening -- ^ We want to specify 'depthWidening' in fixed-point iteration
  -> Id
  -> CoreExpr
  -> (Id, CoreExpr, AnalEnv)
cprAnalBind top_lvl env widening id rhs
  -- See Note [CPR for data structures]
  | isDataStructure id rhs
  = (id,  rhs,  env) -- Data structure => no code => no need to analyse rhs
  | otherwise
  = (id', rhs', env')
  where
    arg_tys             = fst (splitFunNewTys (idType id))
    -- See Note [CPR for binders that will be unboxed]
    -- See Note [Rapid termination for strict binders]
    assumed_arg_cpr_tys = argCprTypesFromStrictSig (unboxingStrategy env)
                                                   arg_tys
                                                   (idStrictness id)

    (rhs_ty, rhs')      = cprAnal env assumed_arg_cpr_tys rhs

    -- possibly trim thunk CPR info
    rhs_ty'
      -- See Note [CPR for thunks]
      | stays_thunk = trimCprTy rhs_ty
      -- See Note [CPR for sum types]
      | returns_sum = trimCprTy rhs_ty
      | otherwise   = rhs_ty

    -- See Note [Arity trimming for CPR signatures]
    -- See Note [Ensuring termination of fixed-point iteration]
    sig    = widening $ mkCprSigForArity (idArity id) rhs_ty'
    id'    = -- pprTrace "cprAnalBind" (ppr id $$ ppr rhs_ty' $$ ppr sig) $
             setIdCprInfo id sig
    env'   = extendSigEnv env id sig

    -- See Note [CPR for thunks]
    stays_thunk = is_thunk && not_strict
    is_thunk    = not (exprIsHNF rhs) && not (isJoinId id)
    not_strict  = not (isStrUsedDmd (idDemandInfo id))
    -- See Note [CPR for sum types]
    (_, ret_ty) = splitPiTys (idType id)
    returns_prod
      | Just (tc, _, _) <- normSplitTyConApp_maybe (ae_fam_envs env) ret_ty
      = isJust (tyConSingleAlgDataCon_maybe tc)
      | otherwise
      = False
    returns_sum = not (isTopLevel top_lvl) && not returns_prod

isDataStructure :: Id -> CoreExpr -> Bool
-- See Note [CPR for data structures]
isDataStructure id rhs =
  idArity id == 0 && exprIsHNF rhs

-- | Returns an expandable unfolding
-- (See Note [exprIsExpandable] in "GHC.Core.Utils") that has
-- So effectively is a constructor application.
cprDataStructureUnfolding_maybe :: Id -> Maybe CoreExpr
cprDataStructureUnfolding_maybe id = do
  -- There are only FinalPhase Simplifier runs after CPR analysis
  guard (activeInFinalPhase (idInlineActivation id))
  unf <- expandUnfolding_maybe (idUnfolding id)
  guard (isDataStructure id unf)
  return unf

unboxingStrategy :: AnalEnv -> UnboxingStrategy Demand
unboxingStrategy env = wantToUnboxArg (ae_fam_envs env) has_inlineable_prag
  where
    -- Rather than maintaining in AnalEnv whether we are in an INLINEABLE
    -- function, we just assume that we are. That flag is only relevant
    -- to Note [Do not unpack class dictionaries], the few unboxing
    -- opportunities on dicts it prohibits are probably irrelevant to CPR.
    has_inlineable_prag = True

--
-- ** Analysing recursive bindings
--

-- | Fixed-point iteration
cprFix
  :: TopLevelFlag
  -> AnalEnv                    -- Does not include bindings for this binding
  -> [(Id,CoreExpr)]
  -> (AnalEnv, [(Id,CoreExpr)]) -- Binders annotated with CPR info
cprFix top_lvl orig_env orig_pairs
  = loop 1 init_env init_pairs
  where
    init_sig id rhs
      -- See Note [CPR for data structures]
      | isDataStructure id rhs = topCprSig
      | otherwise              = mkCprSig (idArity id) divergeCpr
    -- See Note [Initialising strictness] in GHC.Core.Opt.DmdAnal
    orig_virgin = ae_virgin orig_env
    init_pairs | orig_virgin  = [(setIdCprInfo id (init_sig id rhs), rhs) | (id, rhs) <- orig_pairs ]
               | otherwise    = orig_pairs
    init_env = extendSigEnvFromIds orig_env (map fst init_pairs)

    -- The fixed-point varies the idCprInfo field of the binders and and their
    -- entries in the AnalEnv, and terminates if that annotation does not change
    -- any more.
    loop :: Int -> AnalEnv -> [(Id,CoreExpr)] -> (AnalEnv, [(Id,CoreExpr)])
    loop n env pairs
      | found_fixpoint = (reset_env', pairs')
      | otherwise      = -- pprTrace "cprFix:loop" (ppr n <+> ppr (map _prj pairs) <+> ppr (map _prj pairs')) $
                         loop (n+1) env' pairs'
      where
        -- In all but the first iteration, delete the virgin flag
        -- See Note [Initialising strictness] in GHC.Core.Opt.DmdAnal
        (env', pairs') = step (applyWhen (n/=1) nonVirgin env) pairs
        -- Make sure we reset the virgin flag to what it was when we are stable
        reset_env'     = env'{ ae_virgin = orig_virgin }
        found_fixpoint = map (idCprInfo . fst) pairs' == map (idCprInfo . fst) pairs
        _prj (id,_)    = (id, idCprInfo id) -- a helper fun for the trace call

    step :: AnalEnv -> [(Id, CoreExpr)] -> (AnalEnv, [(Id, CoreExpr)])
    step env pairs = mapAccumL go env pairs
      where
        go env (id, rhs) = (env', (id', rhs'))
          where
            (id', rhs', env') = cprAnalBind top_lvl env depthWidening id rhs

{- Note [Arity trimming for CPR signatures]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Although it doesn't affect correctness of the analysis per se, we have to trim
CPR signatures to idArity. Here's what might happen if we don't:

  f x = if expensive
          then \y. Box y
          else \z. Box z
  g a b = f a b

The two lambdas will have a CPR type of @1m@ (so construct a product after
applied to one argument). Thus, @f@ will have a CPR signature of @2m@
(constructs a product after applied to two arguments).
But WW will never eta-expand @f@! In this case that would amount to possibly
duplicating @expensive@ work.

(Side note: Even if @f@'s 'idArity' happened to be 2, it would not do so, see
Note [Don't eta expand in w/w].)

So @f@ will not be worker/wrappered. But @g@ also inherited its CPR signature
from @f@'s, so it *will* be WW'd:

  f x = if expensive
          then \y. Box y
          else \z. Box z
  $wg a b = case f a b of Box x -> x
  g a b = Box ($wg a b)

And the case in @g@ can never cancel away, thus we introduced extra reboxing.
Hence we always trim the CPR signature of a binding to idArity.
-}

data AnalEnv
  = AE
  { ae_sigs   :: SigEnv
  -- ^ Current approximation of signatures for local ids
  , ae_virgin :: Bool
  -- ^ True only on every first iteration in a fixed-point
  -- iteration. See Note [Initialising strictness] in "GHC.Core.Opt.DmdAnal"
  , ae_fam_envs :: FamInstEnvs
  -- ^ Needed when expanding type families and synonyms of product types.
  }

type SigEnv = IdEnv CprSig

instance Outputable AnalEnv where
  ppr (AE { ae_sigs = env, ae_virgin = virgin })
    = text "AE" <+> braces (vcat
         [ text "ae_virgin =" <+> ppr virgin
         , text "ae_sigs =" <+> ppr env ])

emptyAnalEnv :: FamInstEnvs -> AnalEnv
emptyAnalEnv fam_envs
  = AE
  { ae_sigs = emptyVarEnv
  , ae_virgin = True
  , ae_fam_envs = fam_envs
  }

lookupSigEnv :: AnalEnv -> Id -> Maybe CprSig
lookupSigEnv env id = lookupVarEnv (ae_sigs env) id

extendSigEnv :: AnalEnv -> Id -> CprSig -> AnalEnv
extendSigEnv env id sig
  = env { ae_sigs = extendVarEnv (ae_sigs env) id sig }

extendSigEnvList :: AnalEnv -> [(Id, CprSig)] -> AnalEnv
extendSigEnvList env ids_cprs
  = env { ae_sigs = extendVarEnvList (ae_sigs env) ids_cprs }

-- | Extend an environment with the CPR sigs attached to the ids
extendSigEnvFromIds :: AnalEnv -> [Id] -> AnalEnv
extendSigEnvFromIds env ids
  = extendSigEnvList env [ (id, idCprInfo id) | id <- ids ]

nonVirgin :: AnalEnv -> AnalEnv
nonVirgin env = env { ae_virgin = False }

{- Note [Ensuring termination of fixed-point iteration]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fixed-point iteration may fail to terminate for a function like repeat:

  repeat x = x : repeat x

In the first  round, we will infer the Cpr 2(-, -).
In the second round, we will infer the Cpr 2(-, 2(-, -)).
And so on.

Hence it is important to apply a /widening/ operator between iterations to
ensure termination. In the case of DmdAnal, that is simply a check on the
number of iterations, defaulting to Top after a certain limit
(See Note [Safe abortion in the fixed-point iteration] in DmdAnal).
In case of CprAnal, we simply prune Cpr and Termination info after each
iteration to a constant depth of mAX_DEPTH.

Note [CPR for binders that will be unboxed]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If a lambda-bound variable will be unboxed by worker/wrapper (so it must be
demanded strictly), then give it the CPR property. Here's a concrete example
('f1' in test T10482a), assuming h is strict:

  f1 :: Int -> Int
  f1 x = case h x of
          A -> x
          B -> f1 (x-1)
          C -> x+1

If we notice that 'x' is used strictly, we can give it the CPR
property; and hence f1 gets the CPR property too.  It's sound (doesn't
change strictness) to give it the CPR property because by the time 'x'
is returned (case A above), it'll have been evaluated (by the wrapper
of 'h' in the example).

Moreover, if f itself is strict in x, then we'll pass x unboxed to
f1, and so the boxed version *won't* be available; in that case it's
very helpful to give 'x' the CPR property. Otherwise, the worker would
*rebox* 'x' before returning it.

Note that we only want to do this for something that we want to unbox
('wantToUnboxArg'), else we may get over-optimistic CPR results
(e.g. from \x -> x!).

In practice, we derive CPR information directly from the strictness signature
and the argument type in 'cprAnalBind' via 'argCprTypesFromStrictSig'.

Note [Rapid termination for strict binders]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation of any strict binder will rapidly terminate once strictness
worker/wrapper has happened. Here's an example:

  g :: Int -> Int -> (Int, Int)
  g x | x > 0     = (x, x)
      | otherwise = (0, 0)

We want to nestedly unbox the components of the constructed pair, but we
may only do so if we can prove that 'x' terminates rapidly. And indeed it
does! As strictness analysis will tell, 'g' would eval 'x' anyway, so it
is OK to regard 'x' as if it terminates rapidly, because any additional
evaluation beyond the first one will. We get

  g (I# x) = case $wg x of (# a, b #) -> (I# a, I# b)
  $wg x | x ># 0#   = (# x, x #)
        | otherwise = (# 0#, 0# #)

Like for Note [CPR for binders that will be unboxed], we derive
termination information directly from the strictness signature
in 'cprAnalBind' via 'argCprTypesFromStrictSig'.

But in contrast to CPR information, we also have to account for termination of
strict arguments at *call sites* of 'g'! For example

  h :: Int -> ((Int, Int), (Int, Int))
  h z = (g z, g 42)

We saw that 'g' can be unboxed nestedly. And we'd even say that calls to 'g'
itself terminate rapidly, provided its argument 'x' terminates.
Now, can we unbox the pair returned by 'h' nestedly? No! Evaluating
`h (error "boom")` will terminate just fine, but not if we decide to
unbox the first component of 'h'. The key is that we have to uphold
the "provided its argument terminates" precondition at call sites of 'g'.
That clearly is not the case for 'z', which is a lazy binder.

The solution is to "force" 'z' according to the strictness
signature of 'g', which is what 'cprTransformSig' does. It
accounts the TerminationFlag resulting from the forcing to
the termination recorded in the signature, as if there was
a case expression forcing the argument before the call. In
case of 'g', we get that it MightDiverge because forcing of
the argument 'z' MightDiverge.

Why not simply say that 'g' MightDiverge, so that we don't have to be smart at
call sites? Because then we don't get to see that *any* function that uses its
arguments terminates rapidly! In particular, we'd miss to unbox `g 42` above,
which is perfectly within limits; evaluation of `42` terminates rapidly and then
so does the call `g 42`, which allows to unbox the second component of 'h', thus

  h z = case $wh z of
    (# p, a, b #) -> (p, (I# a, I# b))
  $wh z = case $wg 42 of
    (# a, b #) -> (# g z, a, b #)

Note [CPR in a DataAlt case alternative]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In a case alternative, we want to give some of the binders the CPR property.
Specifically

 * The case binder. Inside the alternative, the case binder always has
   the CPR property if the scrutinee has it, meaning that a case on it will
   successfully cancel. Example:
     f x = case x of y { I# x' -> if x' ==# 3
                                     then y
                                     else I# 8 }

   Since 'x' is used strictly, it will be unboxed and given the CPR property
   (See Note [CPR for binders that will be unboxed]).
   By giving 'y' the CPR property, we ensure that 'f' does too, so we get
        f (I# x) = I# (fw x)
        fw x = case x of y { I# x' -> if x' ==# 3 then x' else 8 }

   If the scrutinee has the CPR property, giving it to the case binder will
   never introduce reboxing. We used to be more optimistic before Nested CPR;
   see #19232.

 * The field binders. If the scrutinee had nested CPR information, the field
   binders inherit that information.
   Example (adapted from T10482a):
        f2 t = case t of (x, y)
                 | x<0 -> f2 (MkT2 x (y-1))
                 | y>1 -> 1
                 | otherwise -> x
   Since 'f2' is strict in 't' and even 'x', they will be available unboxed-only.
   Note [CPR for binders that will be unboxed] gives 't' an appropriately nested
   CPR property from which the field binder 'x' inherits its CPR property in turn,
   so that we give 'f2' the CPR property. Implementation in 'extendEnvForDataAlt'.

Note [CPR for sum types]
~~~~~~~~~~~~~~~~~~~~~~~~
TODO: This is out of date since we have join points. See #16570
At the moment we do not do CPR for let-bindings that
   * non-top level
   * bind a sum type
Reason: I found that in some benchmarks we were losing let-no-escapes,
which messed it all up.  Example
   let j = \x. ....
   in case y of
        True  -> j False
        False -> j True
If we w/w this we get
   let j' = \x. ....
   in case y of
        True  -> case j' False of { (# a #) -> Just a }
        False -> case j' True of { (# a #) -> Just a }
Notice that j' is not a let-no-escape any more.

However this means in turn that the *enclosing* function
may be CPR'd (via the returned Justs).  But in the case of
sums, there may be Nothing alternatives; and that messes
up the sum-type CPR.

Conclusion: only do this for products.  It's still not
guaranteed OK for products, but sums definitely lose sometimes.

Note [CPR for thunks]
~~~~~~~~~~~~~~~~~~~~~
If the rhs is a thunk, we usually forget the CPR info, because
it is presumably shared (else it would have been inlined, and
so we'd lose sharing if w/w'd it into a function).  E.g.

        let r = case expensive of
                  (a,b) -> (b,a)
        in ...

If we marked r as having the CPR property, then we'd w/w into

        let $wr = \() -> case expensive of
                            (a,b) -> (# b, a #)
            r = case $wr () of
                  (# b,a #) -> (b,a)
        in ...

But now r is a thunk, which won't be inlined, so we are no further ahead.
But consider

        f x = let r = case expensive of (a,b) -> (b,a)
              in if foo r then r else (x,x)

Does f have the CPR property?  Well, no.

However, if the strictness analyser has figured out (in a previous
iteration) that it's strict, then we DON'T need to forget the CPR info.
Instead we can retain the CPR info and do the thunk-splitting transform
(see WorkWrap.splitThunk).

This made a big difference to PrelBase.modInt, which had something like
        modInt = \ x -> let r = ... -> I# v in
                        ...body strict in r...
r's RHS isn't a value yet; but modInt returns r in various branches, so
if r doesn't have the CPR property then neither does modInt
Another case I found in practice (in Complex.magnitude), looks like this:
                let k = if ... then I# a else I# b
                in ... body strict in k ....
(For this example, it doesn't matter whether k is returned as part of
the overall result; but it does matter that k's RHS has the CPR property.)
Left to itself, the simplifier will make a join point thus:
                let $j k = ...body strict in k...
                if ... then $j (I# a) else $j (I# b)
With thunk-splitting, we get instead
                let $j x = let k = I#x in ...body strict in k...
                in if ... then $j a else $j b
This is much better; there's a good chance the I# won't get allocated.

But what about botCpr? Consider
    lvl = error "boom"
    fac -1 = lvl
    fac 0 = 1
    fac n = n * fac (n-1)
fac won't have the CPR property here when we trim every thunk! But the
assumption is that error cases are rarely entered and we are diverging anyway,
so WW doesn't hurt.

Should we also trim CPR on DataCon application bindings?
See Note [CPR for data structures]!

Note [CPR for data structures]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Long static data structures (whether top-level or not) like

  xs = x1 : xs1
  xs1 = x2 : xs2
  xs2 = x3 : xs3

should not get CPR signatures (#18154), because they

  * Never get WW'd, so their CPR signature should be irrelevant after analysis
    (in fact the signature might even be harmful for that reason)
  * Would need to be inlined/expanded to see their constructed product
  * Recording CPR on them blows up interface file sizes and is redundant with
    their unfolding. In case of Nested CPR, this blow-up can be quadratic!
    Reason: the CPR info for xs1 contains the CPR info for xs; the CPR info
    for xs2 contains that for xs1. And so on.

Hence we don't analyse or annotate data structures in 'cprAnalBind'. To
implement this, the isDataStructure guard is triggered for bindings that satisfy

  (1) idArity id == 0 (otherwise it's a function)
  (2) exprIsHNF rhs   (otherwise it's a thunk, Note [CPR for thunks] applies)

But we can't just stop giving DataCon application bindings the CPR *property*,
for example

  fac 0 = I# 1#
  fac n = n * fac (n-1)

fac certainly has the CPR property and should be WW'd! But FloatOut will
transform the first clause to

  lvl = I# 1#
  fac 0 = lvl

If lvl doesn't have the CPR property, fac won't either. But lvl is a data
structure, and hence (see above) will not have a CPR signature. So instead, when
'cprAnal' meets a variable lacking a CPR signature to extrapolate into a CPR
transformer, 'cprTransform' instead tries to get its unfolding (via
'cprDataStructureUnfolding_maybe'), and analyses that instead.

In practice, GHC generates a lot of (nested) TyCon and KindRep bindings, one
for each data declaration. They should not have CPR signatures (blow up!).

There is a perhaps surprising special case: KindRep bindings satisfy
'isDataStructure' (so no CPR signature), but are marked NOINLINE at the same
time (see the noinline wrinkle in Note [Grand plan for Typeable]). So there is
no unfolding for 'cprDataStructureUnfolding_maybe' to look through and we'll
return topCprType. And that is fine! We should refrain to look through NOINLINE
data structures in general, as a constructed product could never be exposed
after WW.

It's also worth pointing out how ad-hoc this is: If we instead had

    f1 x = x:[]
    f2 x = x : f1 x
    f3 x = x : f2 x
    ...

we still give every function an every deepening CPR signature. But it's very
uncommon to find code like this, whereas the long static data structures from
the beginning of this Note are very common because of GHC's strategy of ANF'ing
data structure RHSs.

Note [CPR for DataCon wrappers]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We treat DataCon wrappers simply by analysing their unfolding. Why not analyse
the unfolding once, upfront? Two reasons:

  1. It's simpler to analyse the unfolding anew at every call site, and the
     unfolding will be pretty cheap to analyse.
  2. The CPR sig we would give the wrapper in 'GHC.Types.Id.Make.mkDataConRep'
     would not take into account whether the arguments to the wrapper had the
     CPR property itself! That would make the CPR transformers derived from
     CPR sigs for DataCon wrappers much less precise than the transformer for
     DataCon workers ('cprTransformDataConWorkSig').
-}
