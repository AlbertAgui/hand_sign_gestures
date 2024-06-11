.DEFAULT_GOAL := help

help:
	@ echo "  make [receipe]"
	@ echo "  receipes:"
	@ echo "  1: install     :install dependencies"
	@ echo "  2: run         :run code"
	@ echo "  3: git_help    :see git commands"

install:
	pip install opencv-python
	pip install mediapipe
	pip install tqdm

git_help:
	@echo " This is a basic set of git instructions usefull to manage the UVM repository"
	@echo " git clone PATH               :Clone git repository"
	@echo " git status                   :Show the working tree status"
	@echo " git add                      :Add file contents to the index"
	@echo " git branch                   :List branches"
	@echo " git branch -d BRANCH         :Erase local branch"
	@echo " git checkout {BRANCH,COMMIT} :Switch to different branch or commit"
	@echo " git checkout -b BRANCH       :Create new branch from current branch"
	@echo " git pull origin BRANCH       :Fetch from and integrate with current branch"
	@echo " git commit -m MESSAGE        :Record changes to the repository"
	@echo " git push origin BRANCH       :Update remote branch with all commits"
	@echo " git push origin -d BRANCH    :Erase remote branch"
	@echo " git log                      :Show commit logs"
	@echo " git stash list               :Show stash stack"
	@echo " git stash [push -m NAME]     :Stash changes into stack. Optionally, choose a name"
	@echo " git stash pop [stash@{n}]    :Recover stashed changes. Optionally, choose a specific stash"
	@echo " git reset --soft HEAD~1      :Undo last commit in local branch"

run:
	python3 hand_track.py