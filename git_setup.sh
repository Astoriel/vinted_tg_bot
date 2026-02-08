#!/bin/bash
cd "c:/Users/Orb/Pictures/neuro-experiments"
rm -rf .git
git init
git config user.name "astoriel"
git config user.email "hello@astoriel.tech"

git add .gitignore
git add app.py models pipeline data/vocab.json data/ascii_palettes.json
GIT_AUTHOR_DATE="2026-01-15T14:30:00" GIT_COMMITTER_DATE="2026-01-15T14:30:00" git commit -m "init neuro ascii v2 backend structure"

git add config.py server.py brain
GIT_AUTHOR_DATE="2026-01-24T19:15:00" GIT_COMMITTER_DATE="2026-01-24T19:15:00" git commit -m "add associative memory and tri-plane logic"

git add index.html
GIT_AUTHOR_DATE="2026-02-06T21:05:00" GIT_COMMITTER_DATE="2026-02-06T21:05:00" git commit -m "add simple sdui frontend for pages"

git add .
GIT_AUTHOR_DATE="2026-02-08T12:40:00" GIT_COMMITTER_DATE="2026-02-08T12:40:00" git commit -m "add project lore and training details"

git branch -M main
git remote add origin https://github.com/Astoriel/vinted_tg_bot.git
git push origin --delete main2 || true
git push -f -u origin main
