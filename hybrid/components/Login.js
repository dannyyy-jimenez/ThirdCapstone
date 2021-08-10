import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, KeyboardAvoidingView, TouchableWithoutFeedback, Keyboard, Image } from 'react-native';
import { Center, Input, Button, Box } from 'native-base';
import * as SecureStore from 'expo-secure-store';
import icon from '../assets/iconion.png';

import API from '../Api'

async function getValueFor(key) {
  let result = await SecureStore.getItemAsync(key);

  return result
}

async function saveKey(key, value) {
  await SecureStore.setItemAsync(key, value);
}

export default function App({navigation}) {
  const [username, setUsername] = React.useState('');
  const [phoneNumber, setPhoneNumber] = React.useState('');
  const [notif_token, setNotifToken] = React.useState('NONE');

  const onLogin = () => {
    API.post('/login', {'username': username, 'phoneNumber': phoneNumber, 'notif_token': notif_token}).then(res => {
      saveKey('PUN', username)
      navigation.navigate('Active', {username})
    }).catch(error => {
      alert("Error!")
    })
  }

  const GetNotificationKey = async() => {
    let notif_key = await SecureStore.getItemAsync("PNT");
    setNotifToken(notif_key)
  }

  React.useEffect(() => {
    GetNotificationKey()
  }, [])

  return (
    <KeyboardAvoidingView behavior="height" style={styles.view}>
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <Center bg="#11212b" style={{width: '100%'}} flex={1}>
          <Image style={{width: 400, height: 400, resizeMode: 'contain'}} source={icon}></Image>
          <Input
            variant="none"
            value={username}
            bg="#0D1A22"
            keyboardAppearance="dark"
            width="80%"
            style={styles.input}
            onChangeText={(value) => setUsername(value)}
            placeholder="Username"
            _light={{
              placeholderTextColor: "blueGray.400",
            }}
            _dark={{
              placeholderTextColor: "blueGray.50",
            }}
          />
          <Input
            variant="none"
            value={phoneNumber}
            bg="#0D1A22"
            keyboardAppearance="dark"
            width="80%"
            style={styles.input}
            keyboardType='numeric'
            onChangeText={(value) => setPhoneNumber(value)}
            placeholder="Phone Number"
            _light={{
              placeholderTextColor: "blueGray.400",
            }}
            _dark={{
              placeholderTextColor: "blueGray.50",
            }}
          />
          <Button style={{backgroundColor: "#0D1A22", marginTop: 20, marginBottom: 100}} _text={{color: "white"}}  _pressed={{opacity: 0.2}} isDisabled={username == ''} onPress={() => onLogin()}>Continue</Button>
        </Center>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  input: {
    marginTop: 8,
    height: 50,
    marginBottom: 8
  },
  view: {
    height: '100%',
    width: '100%',
    flex: 1,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  }
});
