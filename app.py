<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Search</title>
    <style>
      body {
        font-family: 'Roboto', sans-serif;
        margin: 0;
        padding: 0;
        background-color: #eef2f5;
        color: #333;
      }
      header {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 10px 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        z-index: 1000;
      }
      header .container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 20px;
      }
      header .logo {
        font-size: 1.5em;
        font-weight: 700;
      }
      header nav ul {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
      }
      header nav ul li {
        margin-left: 20px;
      }
      header nav ul li a {
        color: #ecf0f1;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
        cursor: pointer;
      }
      header nav ul li a:hover {
        color: #3498db;
      }
      main {
        margin-top: 100px;
      }
      .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
      }
      .search-section, .about-section, .contact-section {
        background-color: #fff;
        padding: 30px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        text-align: center;
      }
      .search-section h2, .about-section h2, .contact-section h2 {
        margin-bottom: 20px;
        font-weight: 500;
        color: #34495e;
      }
      .search-form {
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .search-form label {
        font-size: 1.2em;
        margin-bottom: 10px;
      }
      .search-form input[type="text"] {
        padding: 10px;
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border: 1px solid #ccc;
        border-radius: 8px;
        font-size: 1em;
      }
      .search-form button {
        padding: 10px 20px;
        background-color: #3498db;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s ease;
      }
      .search-form button:hover {
        background-color: #2980b9;
      }
      .results-section {
        margin-top: 40px;
        max-height: calc(100vh - 300px);
        overflow-y: auto;
        padding-bottom: 20px;
      }
      .results-section h2 {
        margin-bottom: 20px;
        font-weight: 500;
        color: #34495e;
      }
      .results-section ul {
        list-style: none;
        padding: 0;
      }
      .results-section ul li {
        background-color: #fff;
        padding: 20px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
      }
      .result-score {
        font-weight: 700;
        margin-bottom: 10px;
      }
      .result-content {
        font-size: 1.1em;
        margin-bottom: 10px;
      }
      .result-metadata {
        font-size: 0.9em;
        color: #666;
      }
      footer {
        background-color: #2c3e50;
        color: #ecf0f1;
        text-align: center;
        padding: 20px 0;
        position: fixed;
        bottom: 0;
        width: 100%;
      }
      footer a {
        color: #3498db;
        text-decoration: none;
        font-weight: 700;
      }
      footer a:hover {
        color: #2980b9;
      }
      footer .container {
        display: flex;
        justify-content: center;
        align-items: center;
      }
      .hidden {
        display: none;
      }
      .contact-form {
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .contact-form label {
        font-size: 1.2em;
        margin-bottom: 10px;
        text-align: left;
        width: 100%;
        max-width: 500px;
      }
      .contact-form input[type="text"], .contact-form input[type="email"], .contact-form textarea {
        padding: 10px;
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border: 1px solid #ccc;
        border-radius: 8px;
        font-size: 1em;
      }
      .contact-form textarea {
        resize: vertical;
        height: 150px;
      }
      .contact-form button {
        padding: 10px 20px;
        background-color: #3498db;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s ease;
      }
      .contact-form button:hover {
        background-color: #2980b9;
      }
      .about-content {
        text-align: left;
        max-width: 800px;
        margin: 0 auto;
      }
      .about-content p {
        font-size: 1.1em;
        line-height: 1.6;
        margin-bottom: 20px;
      }
      .about-content h3 {
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 10px;
      }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <script>
      function showSection(sectionId) {
        document.querySelectorAll('main > .container > section').forEach(section => {
          section.classList.add('hidden');
        });
        document.getElementById(sectionId).classList.remove('hidden');
      }

      document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('header nav ul li a').forEach(link => {
          link.addEventListener('click', event => {
            event.preventDefault();
            showSection(event.target.getAttribute('href').substring(1));
          });
        });
      });
    </script>
  </head>
  <body>
    <header>
      <div class="container">
        <h1 class="logo">Semantic Search</h1>
        <nav>
          <ul>
            <li><a href="#search-section">Home</a></li>
            <li><a href="#about-section">About</a></li>
            <li><a href="#contact-section">Contact</a></li>
          </ul>
        </nav>
      </div>
    </header>
    <main>
      <div class="container">
        <section id="search-section" class="search-section">
          <h2>Find the Information You Need</h2>
          <form method="POST" class="search-form">
            <label for="query">Enter your query:</label>
            <input type="text" id="query" name="query" required placeholder="Type your query here...">
            <button type="submit">Search</button>
          </form>
        </section>
        {% if results %}
          <section class="results-section">
            <h2>Results for "{{ query }}":</h2>
            <ul>
              {% for result in results %}
                <li>
                  <div class="result-score">Score: {{ result.score }}</div>
                  <div class="result-content">{{ result.content }}</div>
                  <div class="result-metadata">{{ result.metadata }}</div>
                </li>
              {% endfor %}
            </ul>
          </section>
        {% endif %}
        <section id="about-section" class="about-section hidden">
          <h2>About</h2>
          <div class="about-content">
            <p>Semantic Search is a tool designed to help you find the information you need quickly and efficiently using advanced search algorithms. Our mission is to provide the most relevant and accurate results for your queries, leveraging state-of-the-art natural language processing technologies.</p>
            <h3>Vision</h3>
            <p>Envision a world where information is easily accessible and users can find precise answers to their questions without sifting through irrelevant data. Semantic Search aims to revolutionize the search experience by understanding the intent behind your queries and delivering the most pertinent results.</p>
            <!-- <h3>Our Team</h3>
            <p>Our team consists of passionate developers, data scientists, and researchers dedicated to improving the way people search for information. We continuously work on enhancing our algorithms and incorporating the latest advancements in artificial intelligence to serve you better.</p> -->
            <h3>Get in Touch</h3>
            <p>If you have any questions, feedback, or suggestions, please feel free to reach out to us through the contact form below. I value your input and look forward to hearing from you!</p>
          </div>
        </section>
        <section id="contact-section" class="contact-section hidden">
          <h2>Contact</h2>
          <form method="POST" class="contact-form">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required placeholder="Your name...">
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required placeholder="Your email...">
            <label for="message">Message:</label>
            <textarea id="message" name="message" required placeholder="Your message..."></textarea>
            <button type="submit">Submit</button>
          </form>
        </section>
      </div>
    </main>
    <footer>
      <!-- <div class="container">
        <p>Made with ❤️ By <a href="https://www.linkedin.com/in/harsh-k04/" target="_blank">Harsh Kumawat</a></p>
        <p>&copy; 2024 Semantic Search. All rights reserved.</p>
      </div> -->
    </footer>
  </body>
</html>
