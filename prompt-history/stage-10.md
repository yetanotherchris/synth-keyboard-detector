1. The detector is detecting the white keys as "width": 15.
2. This is incorrect, they are about 8 pixels. The grey line between them is one pixel.
3. However, the white keys are not 100% white in colour, they might have some grey.
4. This grey is not as dark as the seperator line's grey.
5. So the detector algorithm should find the darkest grey, and use this to separate white keys.
6. It should record in the JSON file the colours it found for each white key.