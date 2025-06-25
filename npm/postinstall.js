#!/usr/bin/env node

/**
 * Post-install script for stsw npm package
 * 
 * This script installs the Python stsw package after npm installation.
 */

import { spawn } from 'child_process';
import { platform } from 'os';

const isWindows = platform() === 'win32';
const pythonCmd = isWindows ? 'python' : 'python3';
const pipCmd = isWindows ? 'pip' : 'pip3';

console.log('Installing stsw Python package...');

// Check Python version
function checkPythonVersion() {
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonCmd, ['--version'], { stdio: 'pipe' });
    
    let output = '';
    proc.stdout.on('data', (data) => output += data);
    proc.stderr.on('data', (data) => output += data);
    
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error('Python is not installed'));
        return;
      }
      
      const match = output.match(/Python (\d+)\.(\d+)/);
      if (match) {
        const major = parseInt(match[1]);
        const minor = parseInt(match[2]);
        
        if (major === 3 && minor >= 9) {
          resolve();
        } else {
          reject(new Error(`Python 3.9+ required, found ${major}.${minor}`));
        }
      } else {
        reject(new Error('Could not parse Python version'));
      }
    });
  });
}

// Install stsw via pip
function installStsw() {
  return new Promise((resolve, reject) => {
    console.log('Running:', pipCmd, 'install stsw');
    
    const proc = spawn(pipCmd, ['install', 'stsw'], {
      stdio: 'inherit',
      shell: isWindows
    });
    
    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Failed to install stsw (exit code: ${code})`));
      }
    });
    
    proc.on('error', (err) => {
      reject(err);
    });
  });
}

// Main installation flow
async function main() {
  try {
    // Check Python version
    console.log('Checking Python version...');
    await checkPythonVersion();
    console.log('✓ Python 3.9+ found');
    
    // Install stsw
    console.log('\nInstalling stsw from PyPI...');
    await installStsw();
    
    console.log('\n✓ stsw successfully installed!');
    console.log('\nYou can now use the stsw command:');
    console.log('  stsw --help');
    console.log('  stsw selftest');
    
  } catch (error) {
    console.error('\n❌ Installation failed:', error.message);
    
    if (error.message.includes('Python')) {
      console.error('\nPlease install Python 3.9 or later:');
      console.error('  • macOS: brew install python3');
      console.error('  • Ubuntu/Debian: sudo apt install python3 python3-pip');
      console.error('  • Windows: Download from https://python.org');
    } else if (error.message.includes('pip')) {
      console.error('\nPlease ensure pip is installed:');
      console.error('  • python3 -m ensurepip --upgrade');
    }
    
    console.error('\nAfter installing dependencies, run:');
    console.error('  npm install -g stsw');
    
    process.exit(1);
  }
}

// Skip in CI environments
if (process.env.CI || process.env.CONTINUOUS_INTEGRATION) {
  console.log('Skipping postinstall in CI environment');
  process.exit(0);
}

main();