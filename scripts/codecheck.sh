#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IVE_ROOT=$(readlink -f $SCRIPT_DIR/../)
cd $IVE_ROOT

set -x
git diff --exit-code -- . ':(exclude).gitmodules' #you should commit before run this script
./clang-format.sh
git diff --exit-code -- . ':(exclude).gitmodules' #return error if file changed