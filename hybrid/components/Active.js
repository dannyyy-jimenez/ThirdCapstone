import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View, Image, TextInput } from 'react-native';
import { NativeBaseProvider, extendTheme, Box, Center, Input, Button } from 'native-base';
import ActionSheet from "react-native-actions-sheet";
import AnimatedLoader from "react-native-animated-loader";
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { useToast } from 'native-base';
import LottieView from 'lottie-react-native';

const actionSheetRef = React.createRef();

import API from '../Api'

const RECORDING_OPTIONS_PRESET_HIGH_QUALITY_CONF = {
  isMeteringEnabled: true,
  android: {
    extension: '.m4a',
    outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_MPEG_4,
    audioEncoder: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_AAC,
    sampleRate: 44100,
    numberOfChannels: 2,
    bitRate: 128000,
  },
  ios: {
    extension: '.wav',
    audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_MAX,
    sampleRate: 44100,
    numberOfChannels: 2,
    bitRate: 128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
};

export default function App({route}) {
  let actionSheet;
  const [contactNumber,  setContactNumber] = React.useState('')
  const [contacts, setContacts] = React.useState([])
  const [recording, setRecording] = React.useState();
  const [sound, setSound] = React.useState();
  const [endangered,  setEndangered] = React.useState(false)
  const [soundLabel, setSoundLabel] = React.useState('Random Noise')
  const toast = useToast()

  async function playSound(uri) {
    console.log('Loading Sound');
    const { sound } = await Audio.Sound.createAsync({uri: uri});
    setSound(sound);

    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      playsInSilentModeIOS: false,
      staysActiveInBackground: true
    });
    console.log('Playing Sound');
    await sound.setVolumeAsync(1)
    await sound.playAsync();
  }

  async function startRecording() {
    try {
      console.log('Requesting permissions..');
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: true
      });
      console.log('Starting recording..');
      const { recording } = await Audio.Recording.createAsync(
         RECORDING_OPTIONS_PRESET_HIGH_QUALITY_CONF
      );
      setRecording(recording);
      console.log('Recording started');
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  }

  React.useEffect(() => {
    if (recording) {
      setTimeout(() => {
        stopRecording()
      }, 3000)
    }
  }, [recording])

  async function stopRecording() {
    if (!recording) {
      const recording = new Audio.Recording();
      console.log(recording)
      console.log(recording.getStatusAsync())
    }
    console.log('Stopping recording..');
    await recording.stopAndUnloadAsync();
    setRecording(undefined);
    const uri = recording.getURI();
    console.log('Recording stopped and stored at', uri);
    // playSound(uri)

    let audio_data = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64
    })

    API.post('/analyze', {audio: audio_data, 'username': route.params.username}).then((res) => {
      console.log(res)
      danger_idxs = [1, 3, 4, 6, 8]
      if (danger_idxs.includes(res.data.idx)) {
        setEndangered(true)
      } else {
        setEndangered(false)
      }
      if (!toast.isActive('toasty')) {
        toast.show({
            id: 'toasty',
            placement: 'top',
            duration: 2000,
            description: res.data.prediction,
            isClosable: false,
            render: () => {
              return (
                <Box bg={danger_idxs.includes(res.data.idx) ? "#E63B2E" : 'teal.400'} _text={{
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: 18
                }} style={{width: 200, display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: 100}} width={400} px={4} py={4} rounded="md" mb={5}>
                  {res.data.prediction}
                </Box>
              )
            },
            status: danger_idxs.includes(res.data.idx) ? 'error' : 'success'
        })
      }
      startRecording()
    }).catch((error) => {

    })
  }

  const formatContactName = (name) => {
    console.log(name)
    if (!name) return ''

    if (!name.match(/^[0-9]+$/)) {
      return name
    }

    var cleaned = ('' + name).replace(/\D/g, '');
    var match = cleaned.match(/^(\d{3})(\d{3})(\d{4})$/);
    if (match) {
      return '(' + match[1] + ') ' + match[2] + '-' + match[3];
    }
    return null;
  }

  const load = () => {
    API.get('/contacts', {'username': route.params.username}).then(res => {
      // update contacts
      setContacts(res.data.contacts)
    }).catch(error => {
      alert("Error!")
    })
  }

  React.useEffect(() => {
    load()
  }, [])

  const onAddContact = () => {
    API.post('/add_contact', {'phone_number': contactNumber, 'username': route.params.username}).then(res => {
      // update contacts
      load()
      setContactNumber('')
    }).catch(error => {
      alert("Error!")
    })
  }

  const onReportDanger = () => {
    setEndangered(true)
    API.post('/danger', {'username': route.params.username}).then(res => {
      if (!toast.isActive('toasty')) {
        toast.show({
            id: 'toasty',
            placement: 'top',
            duration: 2000,
            description: "Reported!",
            isClosable: false,
            render: () => {
              return (
                <Box bg="#E63B2E" _text={{
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: 18
                }} style={{width: 120, display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: 100}} width={400} px={4} py={4} rounded="md" mb={5}>
                  Reported!
                </Box>
              )
            },
            status: 'error'
        })
      }
    }).catch(error => {
      alert("Error!")
    })
  }

  React.useEffect(() => {
    setTimeout(() => {
      startRecording()
    }, 3000)
  }, [])

  return (
    <Center bg="#11212b" flex={1}>

      <View style={{width: '100%', height: 600}}>
        {
          endangered &&
          <LottieView source={require('./danger.json')} autoPlay loop />
        }

        {
          !endangered &&
          <LottieView source={require('./safe.json')} autoPlay loop />
        }
      </View>

      <Button style={{'backgroundColor': '#E63B2E', marginBottom: 10}}  _pressed={{opacity: 0.2}}
        _text={{
          color: "white",
        }} onPress={() => {actionSheetRef.current?.setModalVisible();}}>
        Emergency Contacts
      </Button>
      <Button style={{'backgroundColor': '#E63B2E', marginTop: 10}}  _pressed={{opacity: 0.2}}
        _text={{
          color: "white",
        }} onPress={() => {onReportDanger()}}>
        Report Danger
      </Button>
      <ActionSheet ref={actionSheetRef} containerStyle={{backgroundColor: '#11212b'}} gestureEnabled={true} indicatorColor="#E63B2E">
        <View style={{padding: 2, paddingTop: 15, display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
          <View style={{width: '80%', paddingTop: 5, marginBottom: 50}}>
            <Text style={{color: 'white', fontWeight: 'bold', fontSize: 24, marginBottom: 20}}>Emergency Contacts</Text>
            {
              contacts.map(contact => {
                return (
                  <Text style={{color: 'white', fontWeight: 'bold', fontSize: 18, marginBottom: 10}}>• {formatContactName(contact)}</Text>
                )
              })
            }
          </View>
          <Input
            variant="unstyled"
            keyboardType='numeric'
            value={contactNumber}
            width="80%"
            style={styles.input}
            onChangeText={(value) => setContactNumber(value)}
            bg="#0D1A22"
            keyboardAppearance="dark"
            placeholder="Phone Number"
            _light={{
              placeholderTextColor: "blueGray.400",
            }}
            _dark={{
              placeholderTextColor: "blueGray.50",
            }}
          />
          <Button isDisabled={contactNumber == ''} onPress={() => onAddContact()} style={{backgroundColor: "#0D1A22"}} _text={{color: "white"}}  _pressed={{opacity: 0.2}} width="50%" height="50px">Add Contact</Button>
        </View>
      </ActionSheet>
      {/* <View style={styles.spacer}></View> */}
    </Center>
  );
}

const styles = StyleSheet.create({
  spacer: {
    flex: 1
  },
  input: {
    height: 55,
    marginTop: 10,
    marginBottom: 20
  }
});
