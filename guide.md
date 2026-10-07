If you have any problems let me know sometimes downloading and stuff didnt work right I included the fixes that got it working for me but I couldve forgot one.

Join the huggingface organisation and see if you can see and access the two repos:

```bash
CS4287/fire-smoke-dataset
CS4287/fire-smoke-results
```

You need to install 2 things for git these allow you to use git for more larger objects so images/csvs etc

```bash
brew install git-lfs
brew install git-xet
```

Then initialise both (this does not reinstall them). Do this from the project root too.

```bash
git lfs install
git xet install
```

You can verify if it worked with 

```bash
git lfs version
git xet --version
```

If they don't appear you might need to restart VsCode.

Next you need to create an SSH key for hugging face

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_huggingface
```

Then copy the output from 

```bash
cat ~/.ssh/id_ed25519_huggingface.pub
```

Add that key to:
Hugging face profile -> SSH and GPG Keys -> Add SSH Key

Call it whatever you want I called it git

Create a config for ssh(Im not sure if i added one for my PC if I didnt include this whole file else just include the hf.co bit):

```bash
~/.ssh/config
```

```bash
Host home-pc
  HostName home-pc
  User emma
  IdentityFile ~/.ssh/id_ed25519

Host hf.co
    HostName hf.co
    User git
    IdentityFile ~/.ssh/id_ed25519_huggingface
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
```

Add this to your keychain

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_huggingface
```

You can then test it with

```bash
ssh -T git@hf.co
```

It should output something like

Hi YOUR_USERNAME, welcome to Hugging Face.

Temporarily disable the download of the large files with lfs, then just initalise the hugging face submodules

```bash
export GIT_LFS_SKIP_SMUDGE=1

git submodule update --init --recursive

unset GIT_LFS_SKIP_SMUDGE
```

Downloading the dataset can take a minute so you want to be in a place with decent enough internet but also need to increase the amount of threads downloading stuff with

```bash
git -C data config lfs.concurrenttransfers 16
```

I had a problem where git couldn't verify my SSH key and I dont know if this is a wider problem but on the safer side you can just hard set it with:

```bash
git -C data config core.sshCommand "ssh -i ~/.ssh/id_ed25519_huggingface -o IdentitiesOnly=yes"
```

You can then now fetch the data 

```bash
git -C data lfs fetch --all
```

Then move it into memory, you should see image folder in the data branch start populating when this runs.

```bash
git -C data reset --hard HEAD
```

For the results folder normal git stuff just works so you can just do 

```bash
cd results
git pull
```


