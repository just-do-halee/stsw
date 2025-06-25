#!/usr/bin/env node

/**
 * stsw CLI wrapper for npm
 * 
 * This wrapper ensures that the Python stsw package is installed
 * and forwards all commands to the Python CLI.
 */

import { spawn } from 'child_process';
import { platform } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Check if Python is available
function checkPython() {
  const pythonCmd = platform() === 'win32' ? 'python' : 'python3';
  
  return new Promise((resolve) => {
    const proc = spawn(pythonCmd, ['--version'], { stdio: 'pipe' });
    
    proc.on('close', (code) => {
      resolve(code === 0 ? pythonCmd : null);
    });
    
    proc.on('error', () => {
      resolve(null);
    });
  });
}

// Check if stsw Python package is installed
function checkStswInstalled(pythonCmd) {
  return new Promise((resolve) => {
    const proc = spawn(pythonCmd, ['-c', 'import stsw; print(stsw.__version__)'], { 
      stdio: 'pipe' 
    });
    
    let output = '';
    proc.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    proc.on('close', (code) => {
      resolve(code === 0 ? output.trim() : null);
    });
  });
}

// Main CLI function
async function main() {
  // Check Python availability
  const pythonCmd = await checkPython();
  
  if (!pythonCmd) {
    console.error('Error: Python 3 is required but not found.');
    console.error('Please install Python 3.9 or later from https://python.org');
    process.exit(1);
  }
  
  // Check if stsw is installed
  const stswVersion = await checkStswInstalled(pythonCmd);
  
  if (!stswVersion) {
    console.error('Error: stsw Python package is not installed.');
    console.error('This should have been installed automatically.');
    console.error('Please try reinstalling: npm install -g stsw');
    process.exit(1);
  }
  
  // Forward command to Python stsw
  const args = process.argv.slice(2);
  const proc = spawn(pythonCmd, ['-m', 'stsw.cli', ...args], {
    stdio: 'inherit',
    shell: false
  });
  
  proc.on('close', (code) => {
    process.exit(code || 0);
  });
  
  proc.on('error', (err) => {
    console.error('Error running stsw:', err.message);
    process.exit(1);
  });
}

// Handle errors
process.on('unhandledRejection', (err) => {
  console.error('Unhandled error:', err);
  process.exit(1);
});

// Run main
main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});