# Add additions to `.bashrc`

The following depends on where you clone this git repo.

We assume that we have cloned our the repo to `$HOME/thesis/master-code`.
Add the following to the `.bashrc` (must be in the same order):

```
if [ -f $HOME/thesis/master-code/setup/bashrc.symlinks ]; then
  . $HOME/thesis/master-code/setup/bashrc.symlinks
  . $HOME/thesis/master-code/setup/bashrc.envvars
  . $HOME/thesis/master-code/setup/bashrc.aliases
fi
```

Now log out and back into the server for the changes to take affect.
