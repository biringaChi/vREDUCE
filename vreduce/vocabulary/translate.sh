# path = ? (pass absolute or relative directory)

for file in $path; do
   clang -cc1 -ast-dump $file >> $file.txt
done