{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE TypeFamilies #-}

module Language.Haskell.Syntax.Pat where

import Language.Haskell.Syntax.Extension ( XRec )
import Data.Kind

type role Pat nominal
data Pat (i :: Type)
type LPat i = XRec i (Pat i)
