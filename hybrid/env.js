import Constants from 'expo-constants';

const ENV = {
  dev: {
    baseUrl: 'http://10.1.10.102:8080',
    apiUrl: 'http://10.0.0.102:8080/api'
  },
  staging: {
    baseUrl: 'http://10.0.0.102:80',
    apiUrl: "http://10.0.0.102:80"
  },
  prod: {
    baseUrl: 'https://d1kx3aye6l7gaf.cloudfront.net/',
    apiUrl: "https://d1kx3aye6l7gaf.cloudfront.net/api"
  }
};

function getEnvVars(env = "") {
  if (!env || env === null || env === undefined || env === "") return ENV.dev;
  if (env.indexOf("dev") !== -1) return ENV.dev;
  if (env.indexOf("staging") !== -1) return ENV.staging;
  if (env.indexOf("prod") !== -1) return ENV.prod;
  return ENV.dev;
}

export default getEnvVars(Constants.manifest.releaseChannel);
