#!/usr/bin/env node
// PreToolUse hook (Bash). Denies destructive git commands and asks before
// rm -rf. Heuristic string match, not a shell parser: errs toward flagging.
// See docs/adr/0001-development-workflow.md.

let input = "";
process.stdin.on("data", (c) => (input += c));
process.stdin.on("end", () => {
  let command;
  try {
    command = JSON.parse(input)?.tool_input?.command;
  } catch {
    process.exit(0); // fail open on unparseable payload
  }
  if (typeof command !== "string") process.exit(0);

  const decide = (permissionDecision, reason) => {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          permissionDecision,
          permissionDecisionReason:
            permissionDecision === "deny"
              ? `Blocked by .claude/hooks/block-dangerous-git.mjs: ${reason} Run it manually outside Claude Code if intended.`
              : reason,
        },
      }),
    );
    process.exit(0);
  };

  const push = /\bgit\s+push\b/.test(command);
  if (push && (/--force\b(?!-with-lease)/.test(command) || /(^|\s)-f(\s|$)/.test(command)))
    decide("deny", "git push --force can overwrite remote history.");
  if (push && /--no-verify\b/.test(command))
    decide("deny", "git push --no-verify skips pre-push checks.");
  if (/\bgit\s+commit\b.*--no-verify\b/.test(command))
    decide("deny", "git commit --no-verify skips pre-commit checks.");
  if (/\bgit\s+reset\s+--hard\b/.test(command))
    decide("deny", "git reset --hard discards uncommitted work.");
  if (/\bgit\s+clean\s+-\w*f/.test(command))
    decide("deny", "git clean -f deletes untracked files.");
  if (/\bgit\s+branch\s+-D\b/.test(command))
    decide("deny", "git branch -D force-deletes an unmerged branch.");

  const rmRf = /\brm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*|-r\s+-f|-f\s+-r)\b/.test(command);
  if (rmRf && !/scratchpad|[\\/]tmp[\\/]|AppData[\\/]Local[\\/]Temp/i.test(command))
    decide("ask", "rm -rf outside a temp/scratchpad path - confirm it is intended.");

  process.exit(0);
});
