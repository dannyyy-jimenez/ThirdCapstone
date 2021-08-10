import axios from 'axios';
import qs from 'qs';
import env from './env';

const headers = {
  'content-type': 'application/x-www-form-urlencoded',
  'X-Requested-With': 'XMLHttpRequest'
};

const client = axios.create({
  baseURL: env.apiUrl,
  timeout: 20000
});

const onSuccess = (res) => {
  if (res.data._hE) {
    return {
      isError: true,
      responseCode: res.status,
      response: res.data._e,
      data: res.data._body,
      date: new Date().getTime()
    }
  }

  return {
    isError: false,
    responseCode: res.status,
    response: 'success',
    data: res.data._body,
    date: new Date().getTime()
  }
};

const onError = (error) => {
  if (!error.response) {
    error.response = {
      status: 500
    };
  }
  return {
    isError: true,
    responseCode: error.response.status,
    response: error.response.data ? error.response.data._e : 'error',
    data: null,
    date: new Date().getTime()
  }
};

export default {
  get: async (query, data) => {
    try {
      const res = await client.get(query, {
        params: data,
        headers: headers
      });
      return onSuccess(res);
    } catch (error) {
      return onError(error);
    }
  },
  post: async (uri, data) => {
    try {
      const res = await client({
        method: 'post',
        headers: headers,
        responseType: 'json',
        url: uri,
        data: qs.stringify(data)
      });
      return onSuccess(res);
    } catch (error) {
      return onError(error);
    }
  }
}
