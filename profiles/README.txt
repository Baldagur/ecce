Put CMYK .icc / .icm profiles here (e.g. your print vendor's, Coated FOGRA39,
US Web Coated SWOP...). build.py copies them next to the app (PSDPrep/profiles/);
you can also drop them there later. They are used for CMYK files that have NO
embedded profile, the first one (alphabetical) wins, ahead of any system
profile, and the Original panel shows which one is in use. A file's embedded
profile is always used first.
