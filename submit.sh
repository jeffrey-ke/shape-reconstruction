#!/bin/bash

cp -r . ../proj2
cd ../proj2
rm *.pth
# remove a ton of gifs, basically, all gifs that aren't the ones I use in the md. 
# some kind of rm function.
cd ..
tar -cf proj2.tar proj2
scp proj2.tar jke3@linux.andrew.cmu.edu:/afs/andrew.cmu.edu/usr/jke3/
ssh jke3linux.andrew.cmu.edu
