#!/bin/bash
# Run the kaggle CLI as a named account without keys ever appearing in a command, the chat or the repo.
#   kaggle_as.sh <account> kernels push -p kaggle/unified
#   kaggle_as.sh <account> kernels status <user>/<slug>
# Keys: ~/.kaggle-<account>/kaggle.json (chmod 600, created by the account's owner). "main" is ~/.kaggle/kaggle.json.
set -e
acct="$1"; shift
cfg="$HOME/.kaggle-$acct"; [ "$acct" = main ] && cfg="$HOME/.kaggle"
f="$cfg/kaggle.json"
[ -f "$f" ] || { echo "no key file $f (create it yourself; never paste a key into chat)" >&2; exit 2; }
[ "$(stat -f %Lp "$f" 2>/dev/null || stat -c %a "$f")" = 600 ] || { echo "$f must be chmod 600" >&2; exit 2; }
export KAGGLE_CONFIG_DIR="$cfg"
export KAGGLE_USERNAME="$(python3 -c "import json,sys;print(json.load(open('$f'))['username'])")"
export KAGGLE_KEY="$(python3 -c "import json,sys;print(json.load(open('$f'))['key'])")"
exec kaggle "$@"
