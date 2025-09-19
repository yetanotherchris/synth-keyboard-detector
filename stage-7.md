1. Write all debug jpg files to the ./temp folder
2. I can see in the JSON output that "'key_index": 1,' has '"x": 25,'. However the first key's width is only 7. So this is far beyond X being 8 or 9 as we'd expect.
3. The width of each key and the height should be exactly the same.
4. The main issue is that there's a grey line between each white key, which will offset the following white key's X by a few pixels.
5. The number of expected white keys is the width of the image divided by the white key's width, minus the width in pixels of each grey line between the white keys.