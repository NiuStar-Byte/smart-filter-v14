# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

## 🔄 **CRITICAL: Local/GitHub Sync Protocol**

**RULE:** Local development repo must ALWAYS stay perfectly synced with GitHub. No divergence. No exceptions.

### **Workflow (Non-Negotiable)**
1. **Make changes** → edit files locally
2. **Stage & commit** → `git add` + `git commit`
3. **PUSH IMMEDIATELY** → `git push origin main` (do not wait)
4. **VERIFY SYNC** → `git log -1` vs `git log origin/main -1` (must match)
5. **Only then** → Continue to next task

### **Verification Commands**
```bash
# Check if synced
git fetch origin
git diff HEAD origin/main  # Should be empty

# If diverged, FIX immediately
git status  # Shows how many commits ahead

# NEVER let this happen:
# "Your branch is ahead of 'origin/main' by 10 commits"

# If you see that, push NOW:
git push origin main
```

### **Why This Matters**
- Local is cloned for **faster execution**, not divergence
- Divergence = rework, conflicts, lost time
- GitHub is source of truth
- Push immediately = always in sync = no errors

### **If Divergence Occurs (Emergency Recovery)**
```bash
git fetch origin
git status  # Identify divergence
git push origin main  # Try direct push
# If rejected, use cherry-pick (last resort):
git reset --hard origin/main
git cherry-pick <your-commits>
git push origin main
```

**Never let local get ahead of GitHub without pushing first.**

---

Add whatever helps you do your job. This is your cheat sheet.
