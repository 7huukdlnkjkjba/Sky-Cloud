use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let current_dir = std::env::current_dir().expect("Failed to get current directory");
    println!("Scanning for Python files in: {:?}", current_dir);

    // Find all .py files in the current directory
    let python_files: Vec<_> = fs::read_dir(&current_dir)
        .expect("Failed to read directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "py" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if python_files.is_empty() {
        println!("No Python files found in the current directory.");
        return;
    }

    // Check if PyInstaller is installed
    let pyinstaller_installed = Command::new("pyinstaller")
        .arg("--version")
        .output()
        .is_ok();

    if !pyinstaller_installed {
        println!("PyInstaller is not installed. Installing it now...");
        let install_status = Command::new("pip")
            .arg("install")
            .arg("pyinstaller")
            .status()
            .expect("Failed to install PyInstaller");

        if !install_status.success() {
            eprintln!("Failed to install PyInstaller. Please install it manually.");
            return;
        }
    }

    // Compile each Python file
    for py_file in python_files {
        let file_name = py_file.file_stem().unwrap().to_str().unwrap();
        let output_dir = current_dir.join("dist").join(file_name);
        let binary_path = output_dir.join(file_name);

        println!("Compiling: {:?}", py_file);

        let status = Command::new("pyinstaller")
            .arg("--onefile")  // Single binary
            .arg("--noconsole")  // No console window (for GUI apps)
            .arg("--distpath")
            .arg(&output_dir)
            .arg(&py_file)
            .status()
            .expect("Failed to run PyInstaller");

        if status.success() {
            println!("Successfully compiled: {:?}", py_file);
            println!("Output binary: {:?}", binary_path);

            // Auto-run the compiled binary
            println!("Attempting to run the compiled binary...");
            let run_status = Command::new(&binary_path)
                .status()
                .expect("Failed to run the compiled binary");

            if !run_status.success() {
                eprintln!("Binary execution failed with exit code: {:?}", run_status.code());
            }
        } else {
            eprintln!("Failed to compile: {:?}", py_file);
        }
    }

    // Clean up temporary files
    if Path::new("build").exists() {
        fs::remove_dir_all("build").expect("Failed to remove build directory");
    }
    
    // Clean up .spec files for each compiled file
    for py_file in python_files {
        let file_name = py_file.file_stem().unwrap().to_str().unwrap();
        let spec_file = format!("{}.spec", file_name);
        if Path::new(&spec_file).exists() {
            fs::remove_file(&spec_file).expect("Failed to remove .spec file");
        }
    }

    println!("Done!");
}
