1. Remove any unused test .py files
2. The X coordinate detection is not quite right for the white keys.
3. The issue appears to be the width of the white key includes the grey line in its calculation.
4. This should be removed from the width.
5. The following white key should start after this grey line, so the first indication of white.