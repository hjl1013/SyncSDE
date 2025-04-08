import { useState } from 'react'
import './App.css'
import Project from './components/Project'

function App() {
    const [count, setCount] = useState(0)

    return (
        <div className='App'>
            <Project />
        </div>
    )
}

export default App
