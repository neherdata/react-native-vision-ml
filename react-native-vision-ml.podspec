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

  # Build as static framework for better compatibility
  s.static_framework = true

  s.source_files = "ios/**/*.{h,m,mm,swift}"

  s.dependency "React-Core"
  s.dependency "onnxruntime-objc", "~> 1.23.0"

  # Swift support
  s.swift_version = "5.0"

  # Build settings
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Include onnxruntime-objc headers for ObjC++ wrapper
    'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_ROOT)/onnxruntime-objc/objectivec/include'
  }

  # Enable CoreML acceleration
  s.frameworks = "CoreML", "Accelerate", "UIKit", "CoreGraphics"
end
