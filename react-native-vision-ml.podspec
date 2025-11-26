require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-vision-ml"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "13.0" }
  s.source       = { :git => "https://github.com/neherdata/react-native-vision-ml.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm,swift}"

  s.dependency "React-Core"

  # Enable modular headers for onnxruntime-objc to fix Swift interop
  # This allows Swift code to import onnxruntime without requiring global use_modular_headers!
  s.dependency "onnxruntime-objc", "~> 1.19.0"

  # Swift support
  s.swift_version = "5.0"

  # Configure pod to use modular headers for proper Swift/ObjC bridging
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES'
  }

  # Enable CoreML acceleration
  s.frameworks = "CoreML", "Accelerate", "UIKit", "CoreGraphics"
end
