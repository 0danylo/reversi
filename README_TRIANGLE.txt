To play a game between two bots on the 20x10 triangle board:

Usage:
python3 triangle_server.py <command_for_bot1> <command_for_bot2>

Example:
python3 triangle_server.py "python3 random_triangle_bot.py" "python3 random_triangle_bot.py"

The server enforces a 3-second time limit per move.
The board geometry is:
Rows 0-9.
Row 0: Offset 9, Length 2.
Row 1: Offset 8, Length 4.
...
Row 9: Offset 0, Length 20.

Input to bot:
Left-aligned rows of integers (0=empty, 1=you, 2=opponent).

Output from bot:
"row col" (space separated).
Coordinates are global:
Row: 0-9
Col: 0-19 (must be within valid range for the row).
