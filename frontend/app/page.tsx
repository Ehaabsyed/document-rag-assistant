"use client"
import React, { useState } from 'react'

const page = () => {
  const [answer, setAnswer] = useState('');
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const question = formData.get('question') as string;
    console.log('Question submitted:', question);

    const response = fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    })
      .then((res) => res.json())
      .then((data) => {
        // console.log('Data received from backend:', data);
        console.log('Data received from backend:', data.answer);
        setAnswer(data.answer);
      });

      
  };
  return (
    <div className='min-h-screen bg-white w-full flex flex-col justify-start items-center p-10 text-black '>
      <h2 className='text-4xl font-bold mb-40 text-blue-500'>MY FIRST RAG APPLICATION</h2>

      <form onSubmit={handleSubmit} className="flex items-center">
        <input type="text" name="question" placeholder="Ask a question..." className="border border-gray-700 rounded-md py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500" />
        <input type="submit" value="Submit" className="bg-blue-500 text-white rounded-md py-2 px-4 ml-2 cursor-pointer hover:bg-blue-600" />
      </form>

      <div className='mt-10 w-full max-w-2xl'>
        <h3 className='text-2xl font-semibold mb-4'>Answer:</h3>
        <p className='text-lg'>{answer}</p>
      </div>
      
    </div>
  )
}

export default page