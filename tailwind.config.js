module.exports = {
  purge: ['./dist/*.html', './dist/*.js'],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      height: {
        "100": '35rem'
       },
       fontFamily: {
        "JB": "JetBrains Mono, monospace"
       }
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
