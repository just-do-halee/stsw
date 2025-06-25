#!/usr/bin/env node

/**
 * Test script for stsw npm package
 */

import { spawn } from 'child_process';
import { platform } from 'os';

const pythonCmd = platform() === 'win32' ? 'python' : 'python3';

console.log('Testing stsw npm package...\n');

// Test 1: Check Python
console.log('1. Checking Python installation...');
const pythonTest = spawn(pythonCmd, ['--version'], { stdio: 'inherit' });

pythonTest.on('close', (code) => {
  if (code !== 0) {
    console.error('❌ Python not found');
    process.exit(1);
  }
  console.log('✓ Python found\n');
  
  // Test 2: Check stsw module
  console.log('2. Checking stsw Python module...');
  const moduleTest = spawn(pythonCmd, ['-c', 'import stsw; print(f"stsw {stsw.__version__}")'], {
    stdio: 'inherit'
  });
  
  moduleTest.on('close', (code) => {
    if (code !== 0) {
      console.error('❌ stsw Python module not installed');
      process.exit(1);
    }
    console.log('✓ stsw module found\n');
    
    // Test 3: Run stsw --version
    console.log('3. Testing stsw CLI...');
    const cliTest = spawn('node', ['npm/cli.js', '--version'], {
      stdio: 'inherit'
    });
    
    cliTest.on('close', (code) => {
      if (code !== 0) {
        console.error('❌ stsw CLI failed');
        process.exit(1);
      }
      console.log('✓ stsw CLI works\n');
      
      console.log('All tests passed! ✨');
    });
  });
});