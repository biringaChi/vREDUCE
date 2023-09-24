path=`find $PWD/data/draper/train/ -name "*.c"` 
for file in $path; do
   clang -cc1 -ast-dump $file >> "$file.txt"
done