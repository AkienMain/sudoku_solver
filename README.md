# Download
- ## Set project folder
  - example) C:\sudoku_solver-main
# Set configuration file
- /config.csv
  - corrected_image_size: Length of a side of corrected image
  - debug_level: How often print log
    - 0: None
    - 1: Low frequency
    - 2: High frequency
# Solve Sudoku
- ## Set image
  - Set input image in the "input" folder.
    - example) C:\sudoku_solver-main\input\input_image.png
- ## Run code
  - Open command prompt or PowerShell.
  - Set current directory to this project folder.
    - example)
        ```
        cd C:\sudoku_solver-main
        ```
  - Run this code.
    - example)
        ```
        python sudoku_solver input_image.png
        ```
    - If you run code without input image's name, this program import "sample.png" and solve it.
        ```
        python sudoku_solver
        ```