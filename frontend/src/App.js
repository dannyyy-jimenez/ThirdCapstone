import download from './Download_on_the_App_Store_Badge_US-UK_RGB_blk_092917.svg'
import './App.css';
import Lottie from 'react-lottie-player'
import lottieJson from './safe.json'

function App() {
  return (
    <div className="App">
      <nav className="AppNav">
        <h3 className="header">PHONETIX</h3>
        <div className="spacer"></div>
        <a href="#home" className='AppNavTab'>Like</a>
        <a href="#home" className='AppNavTab'>My</a>
        <a href="#home" className='AppNavTab'>Capstone</a>
        <a href="#home" className='AppNavTab'>Project?</a>
        <a href="https://github.com/dannyyy-jimenez/ThirdCapstone" className='AppNavTab'>GitHub</a>
      </nav>
      <header className="App-header">
        <div className="icon-container" style={{"display": 'flex', justifyContent: 'center', alignItems: 'center',}}>
          <Lottie
                loop
                animationData={lottieJson}
                play
                style={{ width: 550, height: 550, position: 'absolute', opacity: 0.2 }}
              />
          <img src='/icon.png' className="App-logo" alt="logo" />
        </div>
        <h3>
        Danger Reported by Machine Learning
        </h3>
        <p className='download-text'>
          DOWNLOAD FOR FREE
        </p>
        <br/>
        <a href="https://github.com/dannyyy-jimenez/ThirdCapstone">
          <img src={download} alt="download"/>
        </a>
      </header>
    </div>
  );
}

export default App;
