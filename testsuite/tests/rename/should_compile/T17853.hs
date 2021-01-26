{-# LANGUAGE DisambiguateRecordFields #-}
{-# OPTIONS_GHC -Werror=unused-imports #-}
module T17853 where

import qualified T17853A
import qualified T17853A as X (X(..))
import qualified T17853A as Y (Y(..))

main :: IO ()
main = do
    print T17853A.X { X.name = "hello" }
    print T17853A.Y { Y.age = 3 }
