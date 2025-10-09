#!/bin/bash
# tests_to_migrate/ の全ファイルをmdarrayに自動変換

cd tests_to_migrate

for file in *.rs; do
    echo "Processing $file..."
    # ndarray::Array2 -> Tensor
    sed -i '' 's/ndarray::Array2/Tensor/g' "$file"
    # use ndarray -> use mdarray
    sed -i '' 's/use ndarray/use mdarray/g' "$file"
    # array! -> tensor!  
    sed -i '' 's/array!/tensor!/g' "$file"
    # .dim() -> .shape()
    sed -i '' 's/\.dim()/\.shape()/g' "$file"
    # .nrows() -> .shape().0
    sed -i '' 's/\.nrows()/.shape().0/g' "$file"
    # .ncols() -> .shape().1
    sed -i '' 's/\.ncols()/.shape().1/g' "$file"
done

echo "Done!"
