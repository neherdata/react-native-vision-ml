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

  # Build settings for Swift/ObjC interop with onnxruntime-objc
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_ALLOW_NON_MODULAR_INCLUDES_IN_FRAMEWORK_MODULES' => 'YES',
    'SWIFT_INCLUDE_PATHS' => '$(PODS_TARGET_SRCROOT)/ios'
  }

  # User target needs these settings too
  s.user_target_xcconfig = {
    'CLANG_ALLOW_NON_MODULAR_INCLUDES_IN_FRAMEWORK_MODULES' => 'YES'
  }

  # Enable CoreML acceleration
  s.frameworks = "CoreML", "Accelerate", "UIKit", "CoreGraphics"
end
