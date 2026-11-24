declare module '@mkkellogg/gaussian-splats-3d' {
  import type { Camera, Scene, WebGLRenderer } from 'three'

  export interface SplatBuffer {}
  export const SceneRevealMode: { Instant: number }
  export const SplatLoader: {
    loadFromFileData(
      data: ArrayBuffer,
      minimumAlpha: number,
      compressionLevel: number,
      optimizeSplatData: boolean,
    ): Promise<SplatBuffer>
  }
  export class Viewer {
    constructor(options: {
      rootElement: HTMLElement
      threeScene: Scene
      renderer: WebGLRenderer
      camera: Camera
      selfDrivenMode: false
      useBuiltInControls: false
      sharedMemoryForWorkers: false
      gpuAcceleratedSort: false
      dynamicScene: true
      ignoreDevicePixelRatio: true
      sceneRevealMode: number
    })
    addSplatBuffers(
      buffers: SplatBuffer[],
      options: object[],
      finalBuild: boolean,
      showLoadingUI: boolean,
      showLoadingUIForSplatTreeBuild: boolean,
      replaceExisting: boolean,
    ): Promise<void>
    update(): void
    render(): void
    dispose(): Promise<void>
  }
}
