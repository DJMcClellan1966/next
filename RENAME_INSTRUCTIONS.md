# Repository Rename Instructions: next → ML-toolbox

## Steps to Rename Repository

### 1. Rename on GitHub (Required First Step)

1. Go to your repository on GitHub: https://github.com/DJMcClellan1966/next
2. Click on **Settings** (top right of repository page)
3. Scroll down to **Repository name** section
4. Change name from `next` to `ML-toolbox`
5. Click **Rename** button

### 2. Update Local Git Remote

After renaming on GitHub, run these commands locally:

```bash
# Update remote URL
git remote set-url origin https://github.com/DJMcClellan1966/ML-toolbox.git

# Verify
git remote -v
```

### 3. Test Connection

```bash
# Test push (will work if rename was successful)
git push
```

## What's Been Updated

- ✅ README.md - Updated title and description
- ✅ Repository name references updated

## Notes

- All existing commits and history will be preserved
- All branches will be preserved
- The repository URL will change to: https://github.com/DJMcClellan1966/ML-toolbox
- Anyone with the old URL will need to update their remotes
