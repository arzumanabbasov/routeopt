# ATM Route Optimization Project

This project is designed to optimize ATM cash replenishment routes. It utilizes various datasets and mapping tools to determine the most efficient routes, saving time and resources.

## Project Structure

The project directory is organized as follows:

```
routeopt/
│
├── cache/          # Cached data for optimization
├── charts/         # Visualizations and charts
├── datasets/       # Input datasets
├── maps/           # Map files and assets
├── utils/          # Utility scripts and modules
├── venv/           # Virtual environment directory
├── main.py         # Main script to run the project
└── requirements.txt # List of Python dependencies
```

## Installation

### Prerequisites

- **Python 3.9 or higher** is required. Make sure Python is installed and added to your system's PATH.
- **Git** is required to clone the repository.

### Installation on Windows

1. **Clone the Repository**:
    ```sh
    cd Desktop
    git clone https://github.com/arzumanabbasov/routeopt.git
    cd routeopt
    ```

2. **Create a Virtual Environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the Virtual Environment**:
    ```sh
    venv\Scripts\activate
    ```

4. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Prepare Datasets**:
    ```sh
    mkdir datasets
    # Place your dataset files in the datasets directory
    ```

6. **Run the Project**:
    ```sh
    python main.py
    ```

### Installation on Linux and macOS

1. **Clone the Repository**:
    ```bash
    cd ~/Desktop
    git clone https://github.com/arzumanabbasov/routeopt.git
    cd routeopt
    ```

2. **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    ```

3. **Activate the Virtual Environment**:
    - On **Linux** and **macOS**:
        ```bash
        source venv/bin/activate
        ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Prepare Datasets**:
    ```bash
    mkdir datasets
    # Place your dataset files in the datasets directory
    ```

6. **Run the Project**:
    ```bash
    python main.py
    ```

## Usage

Once installed, the project can be run using the command:

```sh
python main.py
```

This will execute the main script, which will use the datasets and map files to optimize the ATM routes.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes. Make sure to follow the coding guidelines and include tests where applicable.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please contact the project maintainer at [here](a.arzuman313@gmail.com).
