# Demo Video

[YouTube](https://www.youtube.com/watch?v=KSHXEJjfQM8)

# Example
### Input
![input/sample.png](https://github.com/AkienMain/sudoku_solver/blob/master/input/sample.png?raw=true)  
### Output
![output/sample.png/output.png](https://github.com/AkienMain/sudoku_solver/blob/master/output/sample.png/output.png?raw=true)  

---

# Download
- ## Set project folder
  - example) C:\sudoku_solver-master
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
    - example) C:\sudoku_solver-master\input\input_image.png
- ## Run code
  - Open command prompt or PowerShell.
  - Set current directory to this project folder.
    - example)
        ```
        cd C:\sudoku_solver-master
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
