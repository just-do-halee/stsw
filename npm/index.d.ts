/**
 * TypeScript definitions for stsw CLI
 */

declare module 'stsw' {
  /**
   * stsw CLI commands
   */
  export interface StswCLI {
    /**
     * Inspect a safetensors file
     * @param file - Path to the safetensors file
     */
    inspect(file: string): Promise<void>;
    
    /**
     * Verify CRC32 checksums
     * @param file - Path to the safetensors file
     */
    verify(file: string): Promise<void>;
    
    /**
     * Convert PyTorch checkpoint to safetensors
     * @param input - Input PyTorch file
     * @param output - Output safetensors file
     * @param options - Conversion options
     */
    convert(input: string, output: string, options?: {
      crc32?: boolean;
      bufferSize?: number;
    }): Promise<void>;
    
    /**
     * Run self-test
     */
    selftest(): Promise<void>;
  }
  
  const stsw: StswCLI;
  export default stsw;
}