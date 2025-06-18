import React from 'react';
import {Navbar} from "reactstrap";

const NavigationBar = () => {
    return (
        <Navbar color="dark" dark container="fluid" className="mt-3">
            <div className="text-center col-12">
                <h3 className="text-light">Skin disease classification app</h3>
            </div>
        </Navbar>
    );
}

export default NavigationBar;