# To generate custom input
```bash
echo "50" | python generate.py > sample.50.txt
```
Here `50` is the no. of rows(and cols) of the matrix to generate  
`sample.50.txt` is the file in which the matrix will be saved 

# Running scripts
`bash script.sh nprocs input_file_path`
```bash
bash q1.sh 4 sample.50.txt
bash q2.sh 4 sample.50.txt
```
