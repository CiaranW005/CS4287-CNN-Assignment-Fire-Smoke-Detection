If you have any problems let me know sometimes downloading and stuff didnt work right I included the fixes that got it working for me but I couldve forgot one.
If you're setting it up on my PC most of the stuff stays the same I've added the linux commands too.

Join the huggingface organisation and see if you can see and access the two repos:

```bash
CS4287/fire-smoke-dataset
CS4287/fire-smoke-results
```

## Git setup

You need to install 2 things for git these allow you to use git for more larger objects so images/csvs etc

```bash
brew install git-lfs
brew install git-xet

For linux:
sudo apt update
sudo apt install git-lfs curl unzip

curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh
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

If they don't appear you might need to restart VsCode/terminal.

## Setup huggingface connection
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
    UseKeychain yes # Include the last line only on Mac
```

Add this to your keychain

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_huggingface

For Linux:
ssh-add ~/.ssh/id_ed25519_huggingface
```

You can then test it with

```bash
ssh -T git@hf.co
```

It should output something like

Hi YOUR_USERNAME, welcome to Hugging Face.

## Downloading the dataset/results

Temporarily disable the download of the large files with lfs, then just initalise the hugging face submodules

If you are on the PC only just run this(Theres a bit more later on what other setup stuff you have to do on the PC):
```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

Else on your laptop you can continue:
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

If the images appear as some error in vscode as like view as Text or binary do this:

```bash
cd data
git xet install
git lfs pull
```

Then results can be pulled using this however there is no data in there yet.

```bash
cd ../results
git lfs pull
```

## PC Setup

If you are working on this project or your FYP on the pc there is a parition called /data where you are also a user on.

cd first into wherever you cloned the repository.

For this project you can remove the data folder.

```bash
rm -r ./data
```

Then this links into the one on a shared folder

```bash
ln -s /data/shared/CS4287/fire-smoke-dataset/data
```

You can the move results to your own data

```bash
mkdir -p /data/emma/CS4287
mv ./results /data/emma/CS4287

ln -s /data/emma/CS4287/results
```

Then the references for the submodule will be off so this needs to be ran

```bash
printf 'gitdir: %s\n' "$PWD/.git/modules/results" > /data/emma/CS4287/results/.git

git --git-dir="$PWD/.git/modules/results" \
    config core.worktree /data/emma/CS4287/results
```

Verify it wored with, if something comes up about data being wrong or something that is fine because its no longer your submodule but mine, aslong as you can see it in the project folder its working as intended.

```bash
git -C results status
git -C results rev-parse --show-toplevel
git -C results rev-parse --git-dir
```

Should get something like:

```bash
/data/USERNAME/CS4287/results
/home/USERNAME/Projects/CS4287-JJ-Project/.git/modules/results
```

