# Publishing this package on GitHub

## Option 1: GitHub Desktop

1. Extract `prototype-pooling-bias-hsi-github.zip`.
2. In GitHub Desktop, choose **File → Add local repository** and select the
   extracted directory.
3. If prompted, choose **create a repository here**.
4. Commit all files with a message such as `Initial reproducibility release`.
5. Select **Publish repository**, choose the desired visibility, and publish.

## Option 2: command line

Run these commands inside the extracted directory after creating an empty
GitHub repository:

```bash
git init
git add .
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Before submission, open the GitHub URL in a private/incognito browser window to
confirm that `README.md` and the source code are publicly accessible.
Do not upload raw `.mat` files, checkpoints, credentials, or server logs.

For anonymous review, use a neutral repository/account name and configure a
non-identifying Git author name and email before the first commit. Git commit
metadata and the hosting account are visible even when source files contain no
author names. Restore author and citation metadata only after anonymous review.
