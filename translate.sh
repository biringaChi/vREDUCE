path=`find $PWD/data/mvd/mvd_asts/ -name "*.c"`
for file in $path; do
   clang -cc1 -ast-dump $file >> "$file.txt"
done