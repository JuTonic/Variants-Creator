import os

for counter in range(20):
    os.system(f'xelatex -shell-escape -output-directory="variants/pdf" -aux-directory="variants/temp" "variants/tex/variant {counter + 1}"')