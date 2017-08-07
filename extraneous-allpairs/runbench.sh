#!/bin/ksh -x
for i in 1 2 3 4 5 6 7 8 9 10
do
j=`expr "$i" \* "56"`
a.out 0 $j
a.out 0 $j
a.out 0 $j
done

for i in 1 2 3 4 5 6 7 8 9 10
do
j=`expr "$i" \* "56"`
a.out 1 $j
a.out 1 $j
a.out 1 $j
done
