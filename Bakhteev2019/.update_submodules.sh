# simple script for updating submodules
# see https://stackoverflow.com/questions/5828324/update-git-submodule-to-latest-commit-on-origin
git submodule foreach git pull origin master
