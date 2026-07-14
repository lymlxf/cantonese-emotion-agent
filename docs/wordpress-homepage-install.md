# WordPress Homepage Installation

Use the two separated source files for their intended purposes. The HTML and CSS should not be combined.

1. In WordPress, edit the homepage with Elementor.
2. Set the page layout to **Elementor Canvas**.
3. Add an Elementor **HTML** widget to the page.
4. Open `docs/wordpress-homepage-body.html` and paste only its contents into the Elementor HTML widget.
5. Open `docs/wordpress-homepage.css` and paste only its CSS contents into **WordPress Additional CSS**.
6. Publish or update the page, then check the desktop, tablet, and mobile previews.

Do not paste the CSS into the Elementor HTML widget. The HTML widget should contain only the markup from `wordpress-homepage-body.html`; WordPress Additional CSS should contain only the styles from `wordpress-homepage.css`.
