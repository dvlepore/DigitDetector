This directory contains scanned handwritten digits.
The process took a set of digits written in individual cells, cropped them and resized them to fit a 32x32 bitmap.
The files are stored as input_N_D_X.type where
 - N represents the specific "user" number.
 - D represents the specific "digit" 0-9
 - X represents one specific instance (should be 10 instances per user per digit)
 - type is either bmp or data

The bmp is an actual bitmap image that you can open in a graphical viewer but it requires a bit more processing to read in (though it isn't hard).
The data is more human readable and easily read with a simple program.
   The format consists of one line WIDTH HEIGHT giving the dimensions of the image (e.g. 32x32)
   This is then followed by HEIGHT rows of WIDTH gray-scale values.
     The values range from 0 to 255 with 0 meaning BLACK and 255 meaning WHITE
The json is a JSON data dump of the bitmap and is more machine readable.
   When loaded it should just be a two-dimensional array.

E.g. input_3_2_4.data is the human readable scan of the 4th occurrence of digit 2 for user 3.

sample_read.py is a sample Python program to read in the JSON file into an array.
