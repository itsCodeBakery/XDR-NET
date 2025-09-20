#!/bin/bash
set -e
cd "/kaggle/working/XDR-NET"
git add -A
git commit -m "Deferred push"
git push origin HEAD
