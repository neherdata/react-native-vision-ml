module.exports = {
  dependency: {
    platforms: {
      ios: {
        podspecPath: './react-native-vision-ml.podspec',
        // Request modular headers for dependencies
        configurations: ['Debug', 'Release'],
      },
    },
  },
};
