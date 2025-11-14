// UI Toggle
const container = document.getElementById("container");
const registerBtn = document.getElementById("register");
const loginBtn = document.getElementById("login");

registerBtn.addEventListener("click", () => {
    container.classList.add("active");
});

loginBtn.addEventListener("click", () => {
    container.classList.remove("active");
});

let db;

// Initialize SQLite DB
initDB();
async function initDB() {
    const SQL = await initSqlJs({
        locateFile: file => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/${file}`
    });

    db = new SQL.Database();

    // Create table if not exists
    db.run(`
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        );
    `);

    console.log("SQLite DB Ready!");
}

// SIGNUP
document.querySelector(".sign-up form").addEventListener("submit", function(e){
    e.preventDefault();

    const name = this.querySelector("input[type=text]").value.trim();
    const email = this.querySelector("input[type=email]").value.trim();
    const password = this.querySelector("input[type=password]").value.trim();

    if(!name || !email || !password){
        alert("All fields are required!");
        return;
    }

    // Check if email exists
    const check = db.exec(`SELECT * FROM users WHERE email='${email}'`);
    if (check.length > 0){
        alert("Email already exists!");
        return;
    }

    // Insert new user
    db.run(`
        INSERT INTO users (name,email,password)
        VALUES ('${name}','${email}','${password}')
    `);

    alert("Registration Successful! Please Login.");
    container.classList.remove("active");
});

// LOGIN
document.querySelector(".sign-in form").addEventListener("submit", function(e){
    e.preventDefault();

    const email = this.querySelector("input[type=email]").value.trim();
    const password = this.querySelector("input[type=password]").value.trim();

    const result = db.exec(
        `SELECT * FROM users WHERE email='${email}' AND password='${password}'`
    );

    if(result.length === 0){
        alert("Invalid Email or Password");
        return;
    }

    alert("Login Successful!");
    window.location.href = "home.html";  // redirect to dashboard
});
